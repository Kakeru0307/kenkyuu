import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.op = nn.Sequential(
            nn.GroupNorm(16, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        )

    def forward(self, x):
        return self.op(x)


class FuseTimestep(nn.Module):
    def __init__(self, channels, d_temb):
        super().__init__()
        self.t_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_temb, channels),
        )
        self.op = ConvBlock(channels, channels)

    def forward(self, x, t):
        t = self.t_proj(t)
        h = x + t[:, :, None, None]
        return self.op(h)


class FuseCond(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.op = ConvBlock(channels, channels)

    def forward(self, x, cond):
        h = x + cond
        return self.op(h)


class ReshapeBlock(nn.Module):
    def __init__(self, in_c: int, out_c: int, in_w: int, out_w: int, in_h: int, out_h: int):
        super().__init__()

        self.w_proj = nn.Upsample(
            size=(out_w, in_h)) if out_w > in_w else nn.AdaptiveAvgPool2d((out_w, in_h))
        self.h_proj = nn.Sequential(
            nn.Linear(in_h, out_h),
            nn.SiLU(),
            nn.Linear(out_h, out_h)
        ) if out_h != in_h else nn.Identity()
        self.c_proj = nn.Conv2d(in_c, out_c, kernel_size=1, padding=0)

    def forward(self, x: torch.Tensor):
        '''
        x: [N, Cin, Win, Hin]
        return: [N, Cout, Wout, Hout]
        '''
        x = self.w_proj(x)
        x = self.h_proj(x)
        h = self.c_proj(x)
        return h


class ResBlockSingleModality(nn.Module):
    def __init__(self, in_channels, out_channels, d_temb):
        super().__init__()
        self.in_conv = ConvBlock(in_channels, out_channels)
        self.fuse_timestep = FuseTimestep(out_channels, d_temb)
        self.skip = nn.Identity() if in_channels == out_channels else nn.Conv2d(
            in_channels, out_channels, kernel_size=1, padding=0)

        self.in_c = in_channels
        self.out_c = out_channels

    def forward(self, x, t):

        h = self.in_conv(x)
        h = self.fuse_timestep(h, t)

        return h + self.skip(x)


class FuseCondSingleModality(nn.Module):
    def __init__(self, x_shape, fuse_hyperscore: bool, hyperscore_shape, fuse_chord: bool,  chord_shape):
        super().__init__()

        self.fuse_hyperscore = fuse_hyperscore
        self.fuse_chord = fuse_chord
        self.init_conv = ConvBlock(x_shape[0], x_shape[0])

        if fuse_hyperscore:

            self.hyperscore_proj = ReshapeBlock(
                hyperscore_shape[0],
                x_shape[0],
                hyperscore_shape[1],
                x_shape[1],
                hyperscore_shape[2],
                x_shape[2],
            )
            self.fuse_hyperscore_op = FuseCond(x_shape[0])
        if fuse_chord:
            self.chord_proj = ReshapeBlock(
                chord_shape[0],
                x_shape[0],
                chord_shape[1],
                x_shape[1],
                chord_shape[2],
                x_shape[2],
            )
            self.fuse_chord_op = FuseCond(x_shape[0])

    def forward(self, x, hyperscore=None, chord=None):

        h = self.init_conv(x)
        if self.fuse_hyperscore:
            h = self.fuse_hyperscore_op(h, self.hyperscore_proj(hyperscore))
        if self.fuse_chord:
            h = self.fuse_chord_op(h, self.chord_proj(chord))
        return h + x


class FuseRhythmDoubleModality(nn.Module):
    def __init__(self, height, channels):
        super().__init__()
        self.rhythm_proj_1 = nn.Sequential(
            nn.Linear(height, height // 2),
            nn.SiLU(),
            nn.Linear(height // 2, 1)
        )
        self.rhythm_proj_2 = nn.Sequential(
            nn.Linear(height, height // 2),
            nn.SiLU(),
            nn.Linear(height // 2, 1)
        )
        self.conv1 = ConvBlock(channels, channels)
        self.conv2 = ConvBlock(channels, channels)

    def forward(self, x1, x2):
        h1 = self.rhythm_proj_1(x1)
        h2 = self.rhythm_proj_2(x2)

        return self.conv1(x1+h2), self.conv2(x2+h1)


class MultiHeadSelfAttentionBlock(nn.Module):
    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        assert channels % num_heads == 0, "channels must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.norm = nn.GroupNorm(16, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)  # q, k, v projection
        self.proj_out = nn.Conv2d(channels, channels, 1)  # output projection

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x)
        qkv = self.qkv(h)  # shape: (B, 3C, H, W)
        q, k, v = qkv.chunk(3, dim=1)  # (B, C, H, W) x 3

        # reshape to (B, num_heads, head_dim, HW)
        def reshape_heads(t):
            return t.reshape(B, self.num_heads, self.head_dim, H * W)

        q = reshape_heads(q)  # (B, nH, dH, HW)
        k = reshape_heads(k)  # (B, nH, dH, HW)
        v = reshape_heads(v)  # (B, nH, dH, HW)

        attn_scores = torch.einsum("bhdn,bhdm->bhnm", q, k)  # (B, nH, HW, HW)
        attn_scores = attn_scores * (self.head_dim ** -0.5)
        attn_weights = torch.softmax(attn_scores, dim=-1)  # attention map

        attn_out = torch.einsum(
            "bhnm,bhdm->bhdn", attn_weights, v)  # (B, nH, dH, HW)
        out = attn_out.reshape(B, C, H, W)  # merge heads
        out = self.proj_out(out)

        return x + out  # residual


class DualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, resolution, d_temb,
                 fuse_hyperscore1, hyperscore1_shape,
                 fuse_chord1, chord1_shape,
                 fuse_hyperscore2, hyperscore2_shape,
                 fuse_rhythm, self_attn, num_heads):
        super().__init__()
        self.res_block1 = ResBlockSingleModality(
            in_channels, out_channels, d_temb)
        self.res_block2 = ResBlockSingleModality(
            in_channels, out_channels, d_temb)

        self.fuse_cond_1 = FuseCondSingleModality(
            (out_channels, resolution, resolution),
            fuse_hyperscore1, hyperscore1_shape,
            fuse_chord1, chord1_shape,
        )
        self.fuse_cond_2 = FuseCondSingleModality(
            (out_channels, resolution, resolution),
            fuse_hyperscore2, hyperscore2_shape,
            False, None,
        )

        self.fuse_rhythm = FuseRhythmDoubleModality(
            resolution, out_channels) if fuse_rhythm else None

        self.self_attn1 = MultiHeadSelfAttentionBlock(
            out_channels, num_heads) if self_attn else None
        self.self_attn2 = MultiHeadSelfAttentionBlock(
            out_channels, num_heads) if self_attn else None

    def forward(self, x1, x2, t, hyperscore1=None, hyperscore2=None, chord=None):
        '''
        x1, x2: of shape [B, C1, W1, H1]
        t: of shape [B, d_temb]
        hyperscore1: if not None, of shape [B, C2, W2, H2]
        hyperscore2: if not None, of shape [B, C2, W2, H2]
        chord: if not None, of shape [B, C3, W3, H3]
        '''

        h1 = self.res_block1(x1, t)
        h2 = self.res_block2(x2, t)
        h1 = self.fuse_cond_1(h1, hyperscore1, chord)
        h2 = self.fuse_cond_2(h2, hyperscore2, None)
        if self.fuse_rhythm:
            h1, h2 = self.fuse_rhythm(h1, h2)
        if self.self_attn1:
            h1 = self.self_attn1(h1)
        if self.self_attn2:
            h2 = self.self_attn2(h2)
        return h1, h2


class DownSample(nn.Module):

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels,
                               3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(channels, channels,
                               3, stride=2, padding=1)

    def forward(self, x1, x2):
        x1 = self.conv1(x1)
        x2 = self.conv2(x2)
        return x1, x2


class UpSample(nn.Module):

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = ConvBlock(channels, channels)
        self.conv2 = ConvBlock(channels, channels)

    def forward(self, x1, x2):
        x1 = F.interpolate(x1, scale_factor=2, mode="nearest")
        x2 = F.interpolate(x2, scale_factor=2, mode="nearest")
        return self.conv1(x1), self.conv2(x2)


class DualBlockSequential(nn.Sequential):
    def forward(self, x1, x2, cond_1, cond_2, cond_chord, t):
        for block in self:
            if isinstance(block, DownSample):
                x1, x2 = block(x1, x2)
            elif isinstance(block, UpSample):
                x1, x2 = block(x1, x2)
            else:
                x1, x2 = block(x1, x2, t, cond_1, cond_2, cond_chord)
        return x1, x2


class DualUNet(nn.Module):

    def __init__(self,
                 x1_shape: tuple[int, int, int], x2_shape: tuple[int, int, int],
                 d_temb: int,
                 hyperscore1_shape: tuple[int, int, int],
                 chord1_shape: tuple[int, int, int],
                 hyperscore2_shape: tuple[int, int, int],
                 base_channel: int,
                 channel_mult: list[int],
                 num_blocks: int,


                 hyperscore1_level: list[int], hyperscore2_level: list[int],
                 chord_level: list[int], fuse_rhythm_level: list[int],
                 self_attn_level: list[int], num_heads: int,
                 N: int, use_d3pm: bool):
        super().__init__()
        self.channel = base_channel
        self.channel_levels = [base_channel * m for m in channel_mult]
        self.d_temb = d_temb
        level_num = len(channel_mult)

        # In Proj
        self.x1_in_proj = nn.Sequential(
            nn.Conv2d(x1_shape[0], base_channel, 3, 1, 1),
            ConvBlock(base_channel, base_channel))
        self.x2_in_proj = nn.Sequential(
            nn.Conv2d(x2_shape[0], base_channel, 3, 1, 1),
            ConvBlock(base_channel, base_channel))

        # Timestep Proj
        self.time_embed = nn.Sequential(
            nn.Linear(self.channel, d_temb),
            nn.SiLU(),
            nn.Linear(d_temb, d_temb),
        )

        # Down blocks
        self.down_blocks = nn.ModuleList()
        self.down_blocks.append(DualBlockSequential(
            DualBlock(
                base_channel, base_channel, x1_shape[0],
                self.d_temb, False, hyperscore1_shape, False, chord1_shape,
                False, hyperscore2_shape, False, False, 4
            )
        ))
        curr_channel = base_channel
        down_channels = [base_channel]

        for i in range(level_num):
            fuse_hyperscore1 = i in hyperscore1_level
            fuse_hyperscore2 = i in hyperscore2_level
            fuse_chord = i in chord_level
            fuse_rhythm = i in fuse_rhythm_level
            self_attn = i in self_attn_level

            curr_res = x1_shape[1] // (2 ** i)
            for _ in range(num_blocks):
                dual_block = DualBlock(
                    curr_channel, self.channel_levels[i], curr_res, self.d_temb,
                    fuse_hyperscore1, hyperscore1_shape,
                    fuse_chord, chord1_shape,
                    fuse_hyperscore2, hyperscore2_shape,
                    fuse_rhythm,
                    self_attn, num_heads,
                )
                self.down_blocks.append(DualBlockSequential(dual_block))
                curr_channel = self.channel_levels[i]
                down_channels.append(curr_channel)
            if i != level_num - 1:
                down_sample = DownSample(
                    curr_channel)
                self.down_blocks.append(DualBlockSequential(down_sample))
                down_channels.append(curr_channel)

        # Mid block
        self.mid_blocks = DualBlockSequential(
            *[DualBlock(
                curr_channel, curr_channel, curr_res, self.d_temb,
                True, hyperscore1_shape,
                False, chord1_shape,
                True, hyperscore2_shape,
                True,
                True, num_heads
            ) for _ in range(num_blocks)]
        )

        # Up blocks
        self.up_blocks = nn.ModuleList([])
        for i in reversed(range(level_num)):
            fuse_hyperscore1 = i in hyperscore1_level
            fuse_hyperscore2 = i in hyperscore2_level
            fuse_chord = i in chord_level
            fuse_rhythm = i in fuse_rhythm_level
            self_attn = i in self_attn_level

            curr_res = x1_shape[1] // (2 ** i)

            for j in range(num_blocks + 1):
                up_blocks = [DualBlock(
                    curr_channel +
                    down_channels.pop(
                    ), self.channel_levels[i], curr_res, self.d_temb,
                    fuse_hyperscore1, hyperscore1_shape,
                    fuse_chord, chord1_shape,
                    fuse_hyperscore2, hyperscore2_shape,
                    fuse_rhythm,
                    self_attn, num_heads,
                )]
                curr_channel = self.channel_levels[i]
                if i != 0 and j == num_blocks:
                    up_sample = UpSample(curr_channel)
                    up_blocks.append(up_sample)
                self.up_blocks.append(DualBlockSequential(*up_blocks))

        # Out Proj
        out_channel_1 = x1_shape[0] if not use_d3pm else x1_shape[0] * N
        out_channel_2 = x2_shape[0] if not use_d3pm else x2_shape[0] * N
        self.x1_out_proj = ConvBlock(
            base_channel, out_channel_1)
        self.x2_out_proj = ConvBlock(
            base_channel, out_channel_2)
        self.N = N
        self.use_d3pm = use_d3pm

    def time_step_embedding(self, time_steps: torch.Tensor, max_period: int = 10000):
        """
        ## Create sinusoidal time step embeddings

        :param time_steps: are the time steps of shape `[batch_size]`
        :param max_period: controls the minimum frequency of the embeddings.
        """

        # $\frac{c}{2}$; half the channels are sin and the other half is cos,
        half = self.channel // 2
        # $\frac{1}{10000^{\frac{2i}{c}}}$
        frequencies = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half
        ).to(device=time_steps.device)
        # $\frac{t}{10000^{\frac{2i}{c}}}$
        args = time_steps[:, None].float() * frequencies[None]
        # $\cos\Bigg(\frac{t}{10000^{\frac{2i}{c}}}\Bigg)$ and $\sin\Bigg(\frac{t}{10000^{\frac{2i}{c}}}\Bigg)$
        return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

    def forward(self, x1, x2, cond_1, cond_2, cond_chord, t):

        # In Proj
        h1 = self.x1_in_proj(x1)
        h2 = self.x2_in_proj(x2)

        # Timestep emb
        t_emb = self.time_step_embedding(t)
        t_emb = self.time_embed(t_emb)

        # Down blocks
        h1s = []
        h2s = []
        for block in self.down_blocks:
            h1, h2 = block(h1, h2, cond_1, cond_2, cond_chord, t_emb)
            h1s.append(h1)
            h2s.append(h2)

        # Mid blocks
        h1, h2 = self.mid_blocks(h1, h2, cond_1, cond_2, cond_chord, t_emb)

        # Up blocks
        for block in self.up_blocks:
            h1 = torch.cat([h1, h1s.pop()], dim=1)
            h2 = torch.cat([h2, h2s.pop()], dim=1)
            h1, h2 = block(h1, h2, cond_1, cond_2, cond_chord, t_emb)

        # Out Proj
        h1 = self.x1_out_proj(h1)
        h2 = self.x2_out_proj(h2)

        if self.use_d3pm:
            h1 = h1.reshape(
                h1.shape[0], -1, self.N, h1.shape[2], h1.shape[3]).permute(0, 1, 3, 4, 2)
            h2 = h2.reshape(
                h2.shape[0], -1, self.N, h2.shape[2], h2.shape[3]).permute(0, 1, 3, 4, 2)

        return h1, h2


if __name__ == "__main__":
    from torchinfo import summary

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    B = 2
    x1_shape = (33, 128, 128)  # C, W, H
    x2_shape = (1, 128, 128)
    d_temb = 64
    hyperscore1_shape = (33, 8, 8)
    chord1_shape = (3, 32, 12)
    hyperscore2_shape = (2, 8, 3)

    model = DualUNet(
        x1_shape=x1_shape,
        x2_shape=x2_shape,
        d_temb=d_temb,
        hyperscore1_shape=hyperscore1_shape,
        chord1_shape=chord1_shape,
        hyperscore2_shape=hyperscore2_shape,
        base_channel=32,
        channel_mult=[1, 2, 4, 8],
        num_blocks=2,
        hyperscore1_level=[2, 3],
        hyperscore2_level=[2, 3],
        chord_level=[1, 2],
        fuse_rhythm_level=[0],
        self_attn_level=[3],
        num_heads=2,
        N=4,
        use_d3pm=True,

    ).to(device)
    # 构造输入
    x1 = torch.randn(B, *x1_shape).to(device)
    x2 = torch.randn(B, *x2_shape).to(device)
    t = torch.randint(0, 1000, (B,)).to(device)

    hyperscore1 = torch.randn(B, *hyperscore1_shape).to(device)
    hyperscore2 = torch.randn(B, *hyperscore2_shape).to(device)
    chord = torch.randn(B, *chord1_shape).to(device)

    # 前向
    h1, h2 = model(x1, x2, hyperscore1, hyperscore2, chord, t)
    print("h1:", h1.shape)
    print("h2:", h2.shape)
