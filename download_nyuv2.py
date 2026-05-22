
import os
import sys
import logging
import numpy as np
from PIL import Image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

OUTPUT_DIR = "./data/nyuv2"


TRAIN_SIZE = 795


def download_nyu_mat(output_path: str) -> str:
    """Download the NYUv2 labeled .mat file (~2.8GB)."""
    import urllib.request

    url = "http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat"
    mat_path = os.path.join(output_path, "nyu_depth_v2_labeled.mat")

    if os.path.exists(mat_path):
        logger.info(f"NYUv2 .mat file already exists at {mat_path}")
        return mat_path

    logger.info(f"Downloading NYUv2 labeled dataset from {url}")
    logger.info("This is ~2.8GB, it may take a while...")

    os.makedirs(output_path, exist_ok=True)

    def _progress(count, block_size, total_size):
        pct = min(100, int(count * block_size * 100 / total_size))
        sys.stdout.write(f"\rDownloading: {pct}%")
        sys.stdout.flush()

    urllib.request.urlretrieve(url, mat_path, reporthook=_progress)
    print()  # newline after progress
    logger.info(f"Download complete: {mat_path}")
    return mat_path


def extract_and_split(mat_path: str, output_dir: str) -> None:
    """Extract images and depth maps from .mat file into train/test splits."""
    import h5py

    logger.info(f"Loading .mat file: {mat_path}")
    with h5py.File(mat_path, 'r') as f:
        images = np.array(f['images'])  
        depths = np.array(f['depths'])  

    
    n_samples = images.shape[3]
    logger.info(f"Found {n_samples} labeled samples")

    
    for split in ['train', 'test']:
        os.makedirs(os.path.join(output_dir, split, 'rgb'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, split, 'depth'), exist_ok=True)

    for i in range(n_samples):

        if i < TRAIN_SIZE:
            split = 'train'
            idx = i
        else:
            split = 'test'
            idx = i - TRAIN_SIZE

        
        rgb = images[:, :, :, i].transpose(2, 1, 0)  
        rgb = (rgb).astype(np.uint8)
        img = Image.fromarray(rgb)
        img.save(os.path.join(output_dir, split, 'rgb', f'{idx:04d}.png'))

        
        depth = depths[:, :, i].T  
        depth = depth.astype(np.float32)
        np.save(os.path.join(output_dir, split, 'depth', f'{idx:04d}.npy'), depth)

        if (i + 1) % 200 == 0:
            logger.info(f"Processed {i+1}/{n_samples} images")

    logger.info(f"Extraction complete: {TRAIN_SIZE} train, {n_samples - TRAIN_SIZE} test")


def main():
    if os.path.exists(os.path.join(OUTPUT_DIR, 'train', 'rgb')) and \
       len(os.listdir(os.path.join(OUTPUT_DIR, 'train', 'rgb'))) > 0:
        n_train = len(os.listdir(os.path.join(OUTPUT_DIR, 'train', 'rgb')))
        n_test = len(os.listdir(os.path.join(OUTPUT_DIR, 'test', 'rgb')))
        logger.info(f"NYUv2 already prepared: {n_train} train, {n_test} test images")
        logger.info(f"Location: {OUTPUT_DIR}")
        return

    
    try:
        import h5py  
    except ImportError:
        logger.error("h5py is required to extract NYUv2. Install with: pip install h5py")
        sys.exit(1)

    mat_path = download_nyu_mat(OUTPUT_DIR)
    extract_and_split(mat_path, OUTPUT_DIR)

    logger.info(f"NYUv2 dataset ready at {OUTPUT_DIR}")
    logger.info("You can delete the .mat file to save ~2.8GB:")
    logger.info(f"  rm {mat_path}")


if __name__ == "__main__":
    main()
