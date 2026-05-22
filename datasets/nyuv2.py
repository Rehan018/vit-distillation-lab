import os
import logging

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from PIL import Image

logger = logging.getLogger(__name__)


class NYUv2DepthDataset(Dataset):
    def __init__(self, root_dir: str, split: str = "train", img_size: int = 224, max_depth: float = 10.0):
        self.root_dir = os.path.join(root_dir, split)
        self.img_size = img_size
        self.max_depth = max_depth

        rgb_dir = os.path.join(self.root_dir, "rgb")
        depth_dir = os.path.join(self.root_dir, "depth")

        if not os.path.exists(rgb_dir):
            raise FileNotFoundError(
                f"NYUv2 RGB directory not found: {rgb_dir}\n"
                f"Run: python download_nyuv2.py"
            )

        self.rgb_paths = sorted([os.path.join(rgb_dir, f) for f in os.listdir(rgb_dir) if f.endswith(('.png', '.jpg'))])
        self.depth_paths = sorted([os.path.join(depth_dir, f) for f in os.listdir(depth_dir) if f.endswith(('.png', '.npy'))])
        assert len(self.rgb_paths) == len(self.depth_paths)

        self.rgb_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.rgb_paths)

    def __getitem__(self, idx):
        rgb = Image.open(self.rgb_paths[idx]).convert("RGB")
        rgb = self.rgb_transform(rgb)

        depth_path = self.depth_paths[idx]
        if depth_path.endswith('.npy'):
            depth = np.load(depth_path).astype(np.float32)
        else:
            depth = np.array(Image.open(depth_path), dtype=np.float32) / 1000.0

        depth = Image.fromarray(depth)
        depth = depth.resize((self.img_size, self.img_size), Image.BILINEAR)
        depth = np.clip(np.array(depth, dtype=np.float32), 0, self.max_depth)
        depth = torch.from_numpy(depth).unsqueeze(0)
        return rgb, depth


def _worker_init_fn(worker_id):
    import random
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_nyuv2_dataloaders(root_dir='./data/nyuv2', batch_size=32, img_size=224, num_workers=4, subset_size=None, seed=42, max_depth=10.0):
    train_dataset = NYUv2DepthDataset(root_dir=root_dir, split="train", img_size=img_size, max_depth=max_depth)
    test_dataset = NYUv2DepthDataset(root_dir=root_dir, split="test", img_size=img_size, max_depth=max_depth)

    if subset_size is not None:
        g = torch.Generator().manual_seed(seed)
        train_indices = torch.randperm(len(train_dataset), generator=g)[:subset_size].tolist()
        test_indices = torch.randperm(len(test_dataset), generator=g)[:min(subset_size, len(test_dataset))].tolist()
        train_dataset = Subset(train_dataset, train_indices)
        test_dataset = Subset(test_dataset, test_indices)

    shuffle_g = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, worker_init_fn=_worker_init_fn, generator=shuffle_g)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, worker_init_fn=_worker_init_fn)

    logger.info(f"NYUv2: {len(train_dataset)} train, {len(test_dataset)} test samples")
    return train_loader, test_loader
