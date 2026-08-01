import torch.nn.functional as F
import torch
import torch.nn as nn
import numpy as np

class D3PM(nn.Module):
    def __init__(
        self,
        x0_model: nn.Module,
        n_T: int,
        num_classes: int = 4,
        forward_type="absorb",
        hybrid_loss_coeff=0.001,
    ) -> None:
        super(D3PM, self).__init__()
        self.x0_model = x0_model

        self.n_T = n_T
        self.hybrid_loss_coeff = hybrid_loss_coeff

        steps = torch.arange(n_T + 1, dtype=torch.float64) / n_T
        alpha_bar = torch.cos((steps + 0.008) / 1.008 * torch.pi / 2)
        self.beta_t = torch.minimum(
            1 - alpha_bar[1:] / alpha_bar[:-
                                          1], torch.ones_like(alpha_bar[1:]) * 0.999
        )

        #self.beta_t = [1 / (self.n_T - t + 1) for t in range(1, self.n_T + 1)]
        self.eps = 1e-6
        self.num_classses = num_classes
        q_onestep_mats = []
        q_mats = []  # these are cumulative

        for beta in self.beta_t:

            if forward_type == "uniform":
                mat = torch.ones(num_classes, num_classes) * beta / num_classes
                mat.diagonal().fill_(1 - (num_classes - 1) * beta / num_classes)
                q_onestep_mats.append(mat)
            elif forward_type == "absorb":
                mat = torch.eye(num_classes) * (1 - beta)
                mat[:, -1] += beta
                q_onestep_mats.append(mat)
            elif forward_type == "custom":
                pass
            else:
                raise NotImplementedError
        q_one_step_mats = torch.stack(q_onestep_mats, dim=0)

        q_one_step_transposed = q_one_step_mats.transpose(
            1, 2
        )  # this will be used for q_posterior_logits

        q_mat_t = q_onestep_mats[0]
        q_mats = [q_mat_t]
        for idx in range(1, self.n_T):
            q_mat_t = q_mat_t @ q_onestep_mats[idx]
            q_mats.append(q_mat_t)
        q_mats = torch.stack(q_mats, dim=0)
        self.logit_type = "logit"

        # register
        self.register_buffer("q_one_step_transposed", q_one_step_transposed)
        self.register_buffer("q_mats", q_mats)

        assert self.q_mats.shape == (
            self.n_T,
            num_classes,
            num_classes,
        ), self.q_mats.shape

    def _at(self, a, t, x):
        # t is 1-d, x is integer value of 0 to num_classes - 1
        bs = t.shape[0]
        t = t.reshape((bs, *[1] * (x.dim() - 1)))
        # out[i, j, k, l, m] = a[t[i, j, k, l], x[i, j, k, l], m]
        return a[t - 1, x, :]

    def q_posterior_logits(self, x_0, x_t, t):
        # if t == 1, this means we return the L_0 loss, so directly try to x_0 logits.
        # otherwise, we return the L_{t-1} loss.
        # Also, we never have t == 0.

        # if x_0 is integer, we convert it to one-hot.
        if x_0.dtype == torch.int64 or x_0.dtype == torch.int32:

            x_0_logits = torch.log(
                torch.nn.functional.one_hot(
                    x_0, self.num_classses) + self.eps
            )
        else:
            x_0_logits = x_0.clone()

        assert x_0_logits.shape == x_t.shape + (self.num_classses,), print(
            f"x_0_logits.shape: {x_0_logits.shape}, x_t.shape: {x_t.shape}"
        )

        # Here, we caclulate equation (3) of the paper. Note that the x_0 Q_t x_t^T is a normalizing constant, so we don't deal with that.

        # fact1 is "guess of x_{t-1}" from x_t
        # fact2 is "guess of x_{t-1}" from x_0

    
        fact1 = self._at(self.q_one_step_transposed, t, x_t)

        softmaxed = torch.softmax(x_0_logits, dim=-1)  # bs, ..., num_classes
        shape = softmaxed.shape
        qmats2 = self.q_mats[t - 2].to(dtype=softmaxed.dtype)
        # bs, num_classes, num_classes
        fact2 = torch.einsum("b...c,bcd->b...d", softmaxed, qmats2)

        out = torch.log(fact1 + self.eps) + torch.log(fact2 + self.eps)
        #out = torch.log(fact2 + self.eps)

        t_broadcast = t.reshape((t.shape[0], *[1] * (x_t.dim())))

        bc = torch.where(t_broadcast == 1, x_0_logits, out)

        return bc

    def vb(self, dist1, dist2):

        # flatten dist1 and dist2
        dist1 = dist1.flatten(start_dim=0, end_dim=-2)
        dist2 = dist2.flatten(start_dim=0, end_dim=-2)

        out = torch.softmax(dist1 + self.eps, dim=-1) * (
            torch.log_softmax(dist1 + self.eps, dim=-1)
            - torch.log_softmax(dist2 + self.eps, dim=-1)
        )
        return out.sum(dim=-1).mean()

    def q_sample(self, x_0, t, noise):
        # forward process, x_0 is the clean input.
        logits = torch.log(self._at(self.q_mats, t, x_0) + self.eps)
        noise = torch.clip(noise, self.eps, 1.0)
        gumbel_noise = -torch.log(-torch.log(noise))
        return torch.argmax(logits + gumbel_noise, dim=-1)

    def model_predict(self, x1, x2, cond1, cond2, cond_chord, t):
        # this part exists because in general, manipulation of logits from model's logit
        # so they are in form of x_t's logit might be independent to model choice.
        # for example, you can convert 2 * N channel output of model output to logit via get_logits_from_logistic_pars
        # they introduce at appendix A.8.

        x1_input = (x1 / (self.num_classses-1))*2.0 - 1.0
        x2_input = (x2 / (self.num_classses-1))*2.0 - 1.0

        predicted_x1_logits, predicted_x2_logits = self.x0_model(
            x1_input, x2_input, cond1, cond2, cond_chord, t)

        return predicted_x1_logits, predicted_x2_logits

    def forward(self, batch_segment, null_condition_prob=0.0):
        """
        Makes forward diffusion x_t from x_0, and tries to guess x_0 value from x_t using x0_model.
        x is one-hot of dim (bs, ...), with int values of 0 to num_classes - 1
        """

        x1 = batch_segment[4].to(torch.long)
        cond1 = batch_segment[3]
        x2 = batch_segment[2].to(torch.long)
        cond2 = batch_segment[1]
        cond_chord = batch_segment[0]

        mask_chord = (torch.rand((cond_chord.shape[0]), device=cond_chord.device) > null_condition_prob).float()
        mask_chord = mask_chord.view(-1, 1, 1, 1)  # 变成 [B,1,...] 方便broadcast

        # 这里用 -1 表示"无条件"的embedding（具体实现看你的模型）
        dropped_chord = -torch.ones_like(cond_chord)
        cond_chord_input = cond_chord * mask_chord + dropped_chord * (1 - mask_chord)

        mask_cond_1 = (torch.rand((cond1.shape[0]), device=cond1.device) > null_condition_prob).float()
        mask_cond_1 = mask_cond_1.view(-1, 1, 1, 1)  # 变成 [B,1,...] 方便broadcast

        # 这里用 -1 表示"无条件"的embedding（具体实现看你的模型）
        dropped_cond_1 = -torch.ones_like(cond1)
        cond1_input = cond1 * mask_cond_1 + dropped_cond_1 * (1 - mask_cond_1)




        t = torch.randint(
            1, self.n_T + 1, (x1.shape[0],), device=x1.device)

        xt1 = self.q_sample(
            x1, t, torch.rand(
                (*x1.shape, self.num_classses), device=x1.device)
        )
        xt2 = self.q_sample(
            x2, t, torch.rand(
                (*x2.shape, self.num_classses), device=x2.device)
        )

        # x_t is same shape as x
        assert xt1.shape == x1.shape, print(
            f"xt1.shape: {xt1.shape}, x1.shape: {x1.shape}"
        )
        assert xt2.shape == x2.shape, print(
            f"xt2.shape: {xt2.shape}, x2.shape: {x2.shape}"
        )

        # we use hybrid loss.

        predicted_x1_logits, predicted_x2_logits = self.model_predict(
            xt1, xt2, cond1_input, cond2, cond_chord_input, t)

        # based on this, we first do vb loss.
        true_q_posterior_logits_1 = self.q_posterior_logits(
            x1, xt1, t)
        pred_q_posterior_logits_1 = self.q_posterior_logits(
            predicted_x1_logits, xt1, t)
        vb_loss_1 = self.vb(true_q_posterior_logits_1,
                            pred_q_posterior_logits_1)

        true_q_posterior_logits_2 = self.q_posterior_logits(
            x2, xt2, t)
        pred_q_posterior_logits_2 = self.q_posterior_logits(
            predicted_x2_logits, xt2, t)
        vb_loss_2 = self.vb(true_q_posterior_logits_2,
                            pred_q_posterior_logits_2)

        predicted_x1_logits = predicted_x1_logits.flatten(
            start_dim=0, end_dim=-2)
        x1 = x1.flatten(start_dim=0, end_dim=-1)
        ce_loss_1 = torch.nn.CrossEntropyLoss()(predicted_x1_logits, x1)

        predicted_x2_logits = predicted_x2_logits.flatten(
            start_dim=0, end_dim=-2)
        x2 = x2.flatten(start_dim=0, end_dim=-1)
        ce_loss_2 = torch.nn.CrossEntropyLoss()(predicted_x2_logits, x2)

        return self.hybrid_loss_coeff * (vb_loss_1 + vb_loss_2) + (ce_loss_1 + ce_loss_2), {
            "vb_loss_1": vb_loss_1.detach().item(),
            "vb_loss_2": vb_loss_2.detach().item(),
            "ce_loss_1": ce_loss_1.detach().item(),
            "ce_loss_2": ce_loss_2.detach().item(),
        }

    def p_sample(self, x1, x2, t, cond1, cond2, cond_chord, noise1, noise2, hyperscore_scale=1.0, chord_scale=1.0, x_inp=None,mask=None):
        if hyperscore_scale == 1.0 and chord_scale == 1.0:
            predicted_x1_logits, predicted_x2_logits = self.model_predict(
            x1, x2, cond1, cond2, cond_chord, t)
        elif hyperscore_scale == 0.0 and chord_scale == 0.0:
            predicted_x1_logits, predicted_x2_logits = self.model_predict(
            x1, x2, -torch.ones_like(cond1), cond2, -torch.ones_like(cond_chord), t)

        elif chord_scale == 1.0:
            pred_null_chord, _ = self.model_predict(
                x1, x2, -torch.ones_like(cond1), cond2, cond_chord, t)
            pred_hyperscore_chord, predicted_x2_logits = self.model_predict(
                x1, x2, cond1, cond2, cond_chord, t)
            
            predicted_x1_logits = pred_null_chord + hyperscore_scale  * (pred_hyperscore_chord  - pred_null_chord)
        else:
        # logits_x = nn(null, null) + hyperscore_scale * (nn(hyperscore, null) - nn(null, null)) + chord_scale * (nn(null, chord) - nn(null, null)) + 
            pred_null_chord, _ = self.model_predict(
                x1, x2, -torch.ones_like(cond1), cond2, cond_chord, t)
            pred_hyperscore_null, _ = self.model_predict(
                x1, x2, cond1, cond2, -torch.ones_like(cond_chord), t)
            pred_null_null, _ = self.model_predict(
                x1, x2, -torch.ones_like(cond1), cond2, -torch.ones_like(cond_chord), t)
            pred_hyperscore_chord, predicted_x2_logits = self.model_predict(
                x1, x2, cond1, cond2, cond_chord, t)
            
            predicted_x1_logits = pred_null_null + hyperscore_scale * (pred_hyperscore_null - pred_null_null) + chord_scale * (pred_null_chord - pred_null_null) \
                                + hyperscore_scale * chord_scale * (pred_hyperscore_chord - pred_hyperscore_null - pred_null_chord + pred_null_null)


        pred_q_posterior_logits_1 = self.q_posterior_logits(
            predicted_x1_logits, x1, t)
        pred_q_posterior_logits_2 = self.q_posterior_logits(
            predicted_x2_logits, x2, t)

        noise1 = torch.clip(noise1, self.eps, 1.0 - self.eps)

        not_first_step = (t != 1).float().reshape(
            (x1.shape[0], *[1] * (x1.dim())))

        gumbel_noise1 = -torch.log(-torch.log(noise1))
        sample = torch.argmax(
            pred_q_posterior_logits_1 + gumbel_noise1 * not_first_step, dim=-1
        )
        sample_1 = sample

        
        if isinstance(x_inp, torch.Tensor) and isinstance(mask, torch.Tensor):
            
            x_t_minus_1 = self.q_sample(x_inp, (t-2).to(torch.long), torch.rand(
                (*x_inp.shape, self.num_classses), device=x_inp.device))
            if t.reshape(1).item() == 1:
                x_t_minus_1 = x_inp
            
            sample_1 = (x_t_minus_1 * mask + sample_1 * (1-mask)).to(torch.long)


        noise2 = torch.clip(noise2, self.eps, 1.0 - self.eps)
        gumbel_noise2 = -torch.log(-torch.log(noise2))
        sample = torch.argmax(
            pred_q_posterior_logits_2 + gumbel_noise2 * not_first_step, dim=-1
        )
        sample_2 = sample

        return sample_1, sample_2

    @torch.no_grad()
    def sample(self, x1, x2, cond1, cond2, cond_chord, chord_scale=1.0, hyperscore_scale=1.0, x_inp=None, mask=None, return_stride=10, return_all=False):
        from tqdm import tqdm
        import time
        x1s = []
        x2s = []
        cnt = 0
        for t in tqdm(reversed(range(1, self.n_T + 1))):
            
            t = torch.tensor([t] * x1.shape[0], device=x1.device).to(torch.long)

            B, C1, W, H = x1.shape
            B, C2, W, H = x2.shape
            torch.manual_seed(time.time_ns())
            noise1 = torch.rand(
                (B, C1, W, H, self.num_classses), device=x1.device)
            noise2 = torch.rand(
                (B, C2, W, H, self.num_classses), device=x2.device)

            x1, x2 = self.p_sample(
                x1, x2, t, cond1, cond2, cond_chord, noise1, noise2, 
                chord_scale=chord_scale, hyperscore_scale=hyperscore_scale,
                x_inp=x_inp, mask=mask
            )

            cnt += 1
            if cnt % return_stride == 0 and return_all:
                x1s.append(x1)
                x2s.append(x2)
        if return_all:
            return x1, x2, x1s, x2s
        else:
            return x1, x2
