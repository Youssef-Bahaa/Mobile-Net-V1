import argparse
import os

import torch
import torch.nn as nn
import yaml

from data.imagenet import get_val_loader
from models.MobileNetV1 import MobileNetV1
from train import run_epoch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--checkpoint", default=None,
                        help="defaults to <save_dir>/best_model.pth")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    checkpoint = args.checkpoint or os.path.join(cfg["checkpoint"]["save_dir"], "best_model.pth")

    val_loader = get_val_loader(
        root=cfg["data"]["root"],
        batch_size=cfg["training"]["batch_size"],
        num_workers=cfg["data"]["num_workers"],
        rho=cfg["model"]["rho"],
    )

    model = MobileNetV1(
        num_classes=cfg["model"]["num_classes"], alpha=cfg["model"]["alpha"]
    ).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))

    loss, top1, top5 = run_epoch(model, val_loader, nn.CrossEntropyLoss(), device)
    print(f"alpha={cfg['model']['alpha']} rho={cfg['model']['rho']} | "
          f"loss {loss:.4f} | top-1 {top1:.2f}% | top-5 {top5:.2f}%")


if __name__ == "__main__":
    main()