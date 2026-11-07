from torch.utils.data import DataLoader
from torchvision import datasets, transforms

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def _build_transforms(rho):
    image_size = round(224 * rho)

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.25, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD)
    ])

    val_transform = transforms.Compose([
        transforms.Resize(round(image_size * 256 / 224)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD)
    ])

    return train_transform, val_transform


def get_val_loader(root, batch_size=256, num_workers=8, rho=1.0):
    _, val_transform = _build_transforms(rho)

    val_dataset = datasets.ImageFolder(
        root=f"{root}/val",
        transform=val_transform
    )

    return DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0
    )


def get_imagenet_loaders(root, batch_size=256, num_workers=8, rho=1.0):
    train_transform, _ = _build_transforms(rho)

    train_dataset = datasets.ImageFolder(
        root=f"{root}/train",
        transform=train_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0
    )

    val_loader = get_val_loader(root, batch_size, num_workers, rho)

    return train_loader, val_loader