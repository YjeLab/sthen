"""Build STHEN and run a dummy forward pass (no data required)."""
import torch

from Models.Our.model import HASTFormer
from model_configs import all_configs


def build(name: str) -> HASTFormer:
    cfg = all_configs["OurModel"][name]
    return HASTFormer(
        in_channel=cfg["in_channel"],
        d_model=cfg["hidden_channel"],
        his_len=cfg["seq_in"],
        pre_len=cfg["seq_out"],
        num_heads=4,
        num_layers=cfg["main_layers"],
        in_h=cfg["hight"],
        in_w=cfg["width"],
        nodes=cfg["nodes"],
        data_type=cfg["data_type"],
        patch_size=cfg["patch_size"],
    )


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for name in ("BikeNYC", "PEMS03"):
        model = build(name).to(device).eval()
        cfg = all_configs["OurModel"][name]
        if cfg["data_type"] == "Grid":
            x = torch.rand(2, cfg["seq_in"], cfg["in_channel"], cfg["hight"], cfg["width"], device=device)
        else:
            x = torch.rand(2, cfg["seq_in"], cfg["nodes"], cfg["in_channel"], device=device)
        with torch.no_grad():
            y = model(x)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"{name} [{cfg['data_type']}] params={n_params:,} in={tuple(x.shape)} out={tuple(y.shape)}")


if __name__ == "__main__":
    main()