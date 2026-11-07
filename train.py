import os

import torch
import torch.nn as nn
import yaml
from tqdm import tqdm

from data.imagenet import get_imagenet_loaders
from models.MobileNetV1 import MobileNetV1


def build_optimizer(model, cfg):
    decay, no_decay = [], []
    for m in model.modules():
        for p in m.parameters(recurse=False):
            depthwise = isinstance(m, nn.Conv2d) and m.groups > 1 and m.groups == m.in_channels
            (no_decay if (p.ndim <= 1 or depthwise) else decay).append(p)

    return torch.optim.RMSprop(
        [{"params": decay, "weight_decay": cfg["weight_decay"]},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=cfg["learning_rate"],
        alpha=cfg["rmsprop_alpha"],
        eps=cfg["rmsprop_eps"],
        momentum=cfg["momentum"],
    )


def run_epoch(model, loader, criterion, device, optimizer=None):
    train = optimizer is not None
    model.train(train)
    loss_sum, top1, top5, n = 0.0, 0, 0, 0

    with torch.set_grad_enabled(train):
        for x, y in tqdm(loader, desc="train" if train else "val"):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            out = model(x)
            loss = criterion(out, y)

            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            hit = out.topk(5, dim=1).indices.eq(y[:, None])
            top1 += hit[:, 0].sum().item()
            top5 += hit.any(dim=1).sum().item()
            loss_sum += loss.item() * y.size(0)
            n += y.size(0)

    return loss_sum / n, 100 * top1 / n, 100 * top5 / n


def main():
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)
    t = cfg["training"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_loader, val_loader = get_imagenet_loaders(
        root=cfg["data"]["root"],
        batch_size=t["batch_size"],
        num_workers=cfg["data"]["num_workers"],
        rho=cfg["model"]["rho"],
    )

    model = MobileNetV1(
        num_classes=cfg["model"]["num_classes"], alpha=cfg["model"]["alpha"]
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(model, t)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=t["lr_decay"])

    os.makedirs(cfg["checkpoint"]["save_dir"], exist_ok=True)
    best = 0.0

    for epoch in range(t["epochs"]):
        tr_loss, tr1, tr5 = run_epoch(model, train_loader, criterion, device, optimizer)
        scheduler.step()
        va_loss, va1, va5 = run_epoch(model, val_loader, criterion, device)

        print(f"epoch {epoch + 1}/{t['epochs']} | lr {scheduler.get_last_lr()[0]:.5f} | "
              f"train loss {tr_loss:.4f} top1 {tr1:.2f} top5 {tr5:.2f} | "
              f"val loss {va_loss:.4f} top1 {va1:.2f} top5 {va5:.2f}")

        if va1 > best:
            best = va1
            torch.save(model.state_dict(),
                       os.path.join(cfg["checkpoint"]["save_dir"], "best_model.pth"))

    print(f"best val top-1: {best:.2f}%")


if __name__ == "__main__":
    main()