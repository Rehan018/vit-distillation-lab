import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, Subset
import torch


def _worker_init_fn(worker_id: int) -> None:

    import numpy as np
    import random
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_oxford_pets_dataloaders(
    root_dir: str = './data',
    batch_size: int = 32,
    img_size: int = 224,
    num_workers: int = 4,
    subset_size: int | None = None,
    seed: int = 42,
):
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.CenterCrop((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = torchvision.datasets.OxfordIIITPet(root=root_dir, split='trainval', transform=train_transform, download=True)
    val_dataset = torchvision.datasets.OxfordIIITPet(root=root_dir, split='test', transform=val_transform, download=True)

    if subset_size is not None:
        g = torch.Generator().manual_seed(seed)
        train_indices = torch.randperm(len(train_dataset), generator=g)[:subset_size].tolist()
        val_indices = torch.randperm(len(val_dataset), generator=g)[:subset_size].tolist()
        train_dataset = Subset(train_dataset, train_indices)
        val_dataset = Subset(val_dataset, val_indices)


    shuffle_g = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, worker_init_fn=_worker_init_fn,
        generator=shuffle_g,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, worker_init_fn=_worker_init_fn,
    )

    num_classes = 37  
    return train_loader, val_loader, num_classes
