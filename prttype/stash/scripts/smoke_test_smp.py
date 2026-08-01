"""segmentation-models-pytorch の動作確認。"""

import torch

from model import build_unet


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_unet(in_channels=11, out_channels=11).to(device)
    dummy = torch.randn(1, 11, 128, 128, device=device)

    with torch.no_grad():
        output = model(dummy)

    assert output.shape == (1, 11, 128, 128), f"unexpected shape: {output.shape}"
    print("smp U-Net forward OK")
    print(f"device: {device}")
    print(f"input:  {tuple(dummy.shape)}")
    print(f"output: {tuple(output.shape)}")


if __name__ == "__main__":
    main()
