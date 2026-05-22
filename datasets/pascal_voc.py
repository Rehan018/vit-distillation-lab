import os
import logging
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import VOCSegmentation
from PIL import Image

logger = logging.getLogger(__name__)
NUM_CLASSES = 21
IGNORE_INDEX = 255


class VOCSegmentationTransform:
    def __init__(self, img_size=224):
        self.img_size = img_size
        self.img_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __call__(self, img, mask):
        img = self.img_transform(img)
        mask = mask.resize((self.img_size, self.img_size), Image.NEAREST)
        mask = torch.from_numpy(np.array(mask)).long()
        return img, mask


class VOCWithJointTransform(VOCSegmentation):
    def __init__(self, root, image_set, img_size=224, download=True):
        super().__init__(root=root, image_set=image_set, download=download)
        self.joint_transform = VOCSegmentationTransform(img_size=img_size)

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert("RGB")
        mask = Image.open(self.masks[index])
        img, mask = self.joint_transform(img, mask)
        return img, mask


def _worker_init_fn(worker_id):
    import random
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_voc_dataloaders(root_dir='./data', batch_size=16, img_size=224, num_workers=4, subset_size=None, seed=42):
    train_dataset = VOCWithJointTransform(root=root_dir, image_set='train', img_size=img_size, download=True)
    val_dataset = VOCWithJointTransform(root=root_dir, image_set='val', img_size=img_size, download=True)

    if subset_size is not None:
        g = torch.Generator().manual_seed(seed)
        train_indices = torch.randperm(len(train_dataset), generator=g)[:subset_size].tolist()
        val_indices = torch.randperm(len(val_dataset), generator=g)[:min(subset_size, len(val_dataset))].tolist()
        train_dataset = Subset(train_dataset, train_indices)
        val_dataset = Subset(val_dataset, val_indices)

    shuffle_g = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, worker_init_fn=_worker_init_fn, generator=shuffle_g)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, worker_init_fn=_worker_init_fn)

    logger.info(f"Pascal VOC: {len(train_dataset)} train, {len(val_dataset)} val samples")
    return train_loader, val_loader, NUM_CLASSES
