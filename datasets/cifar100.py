import logging

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

logger = logging.getLogger(__name__)

NUM_CLASSES = 100


def _worker_init_fn(worker_id: int) -> None:
    import random
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_cifar100_dataloaders(
    root_dir: str = './data',
    batch_size: int = 64,
    img_size: int = 224,
    num_workers: int = 4,
    subset_size: int | None = None,
    seed: int = 42,
):
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                             std=[0.2675, 0.2565, 0.2761]),
    ])

    test_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                             std=[0.2675, 0.2565, 0.2761]),
    ])

    train_dataset = datasets.CIFAR100(
        root=root_dir, train=True, transform=train_transform, download=True,
    )
    test_dataset = datasets.CIFAR100(
        root=root_dir, train=False, transform=test_transform, download=True,
    )

    if subset_size is not None:
        g = torch.Generator().manual_seed(seed)
        train_indices = torch.randperm(len(train_dataset), generator=g)[:subset_size].tolist()
        test_indices = torch.randperm(len(test_dataset), generator=g)[:min(subset_size, len(test_dataset))].tolist()
        train_dataset = Subset(train_dataset, train_indices)
        test_dataset = Subset(test_dataset, test_indices)

    shuffle_g = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, worker_init_fn=_worker_init_fn,
        generator=shuffle_g,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, worker_init_fn=_worker_init_fn,
    )

    logger.info(f"CIFAR-100: {len(train_dataset)} train, {len(test_dataset)} test samples")
    return train_loader, test_loader, NUM_CLASSES
