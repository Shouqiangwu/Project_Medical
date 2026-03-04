"""
NIH Chest X-ray 14 data loading and single-label filtering.

Supports:
  - Kaggle 224x224 pre-resized version (primary)
  - Original 1024x1024 images with on-the-fly resize (fallback)

Target 6 classes (single-label only):
  0: No Finding
  1: Atelectasis
  2: Cardiomegaly
  3: Effusion
  4: Infiltration
  5: Mass
"""
import os
import random
from typing import Dict, List, Optional

import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T


# ======================== Label Mapping ========================
LABEL_NAMES = [
    "No Finding",     # 0
    "Atelectasis",    # 1
    "Cardiomegaly",   # 2
    "Effusion",       # 3
    "Infiltration",   # 4
    "Mass",           # 5
]
LABEL_TO_IDX = {name: i for i, name in enumerate(LABEL_NAMES)}
NUM_CLASSES = 6


# ======================== Dataset Class ========================
class NIHChestXrayDataset(Dataset):
    """
    PyTorch Dataset for filtered NIH CXR-14 (single-label, 6 classes).

    Args:
        img_dir:  path to the folder containing .png images
        csv_path: path to Data_Entry_2017.csv (or _v2020.csv)
        transform: torchvision transform to apply
        max_per_class: cap per class (for balancing "No Finding")
        split_file: optional path to a txt file listing image filenames for this split
    """

    def __init__(
        self,
        img_dir: str,
        csv_path: str,
        transform=None,
        max_per_class: Optional[int] = None,
        split_file: Optional[str] = None,
        seed: int = 2025,
    ):
        self.img_dir = img_dir
        self.transform = transform

        # 1. Load metadata CSV
        df = pd.read_csv(csv_path)

        # 2. If a split file is provided, filter to those images
        if split_file is not None and os.path.exists(split_file):
            with open(split_file, "r") as f:
                split_names = set(line.strip() for line in f if line.strip())
            df = df[df["Image Index"].isin(split_names)].reset_index(drop=True)

        # 3. Filter to single-label images in our 6 target classes
        df["label_list"] = df["Finding Labels"].str.split("|")
        df["num_labels"] = df["label_list"].apply(len)
        mask = (df["num_labels"] == 1) & (df["Finding Labels"].isin(LABEL_TO_IDX))
        df = df[mask].reset_index(drop=True)

        # 4. Map string labels to integer indices
        df["label_idx"] = df["Finding Labels"].map(LABEL_TO_IDX)

        # 5. Class balancing: cap each class
        if max_per_class is not None:
            rng = random.Random(seed)
            balanced_rows = []
            for c in range(NUM_CLASSES):
                class_df = df[df["label_idx"] == c]
                if len(class_df) > max_per_class:
                    sampled = class_df.sample(n=max_per_class, random_state=seed)
                    balanced_rows.append(sampled)
                else:
                    balanced_rows.append(class_df)
            df = pd.concat(balanced_rows, ignore_index=True)

        # 6. Store final data
        self.filenames = df["Image Index"].tolist()
        self.targets = df["label_idx"].tolist()       # list[int], needed by build_class_index

        print(f"[NIHChestXrayDataset] Loaded {len(self)} images across {NUM_CLASSES} classes")
        for c in range(NUM_CLASSES):
            cnt = self.targets.count(c)
            print(f"  Class {c} ({LABEL_NAMES[c]}): {cnt} images")

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]
        label = self.targets[idx]

        img_path = os.path.join(self.img_dir, fname)
        img = Image.open(img_path).convert("L")   # force grayscale

        if self.transform:
            img = self.transform(img)

        return img, label


# ======================== Main Entry Point ========================
def get_nih_train_and_test(
    data_dir: str,
    max_per_class: Optional[int] = 3000,
    image_size: int = 224,
):
    """
    Load NIH CXR-14 training and test sets.

    Expected directory layout (Kaggle 224x224 version):
        data_dir/
            images_224/          (or images/)    <- all .png files
            Data_Entry_2017.csv                  <- metadata
            train_val_list.txt                   <- official train split
            test_list.txt                        <- official test split

    Returns:
        train_raw:  Dataset, images in [0,1], shape (1, 224, 224)
        test_norm:  Dataset, images normalized to [-1,1]
        mean:       tuple, normalization mean
        std:        tuple, normalization std
    """
    # MedVAE convention: normalize to [-1, 1]
    mean = (0.5,)
    std = (0.5,)

    # Auto-detect image subfolder
    img_dir = _find_image_dir(data_dir)
    csv_path = _find_csv(data_dir)
    train_list = _find_file(data_dir, "train_val_list.txt")
    test_list = _find_file(data_dir, "test_list.txt")

    # Transforms
    raw_tf = T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),                          # -> [1, H, W] in [0, 1]
    ])

    test_tf = T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(mean, std),                # -> [-1, 1]
    ])

    train_raw = NIHChestXrayDataset(
        img_dir=img_dir,
        csv_path=csv_path,
        transform=raw_tf,
        max_per_class=max_per_class,
        split_file=train_list,
    )

    test_norm = NIHChestXrayDataset(
        img_dir=img_dir,
        csv_path=csv_path,
        transform=test_tf,
        max_per_class=None,            # no cap for test set
        split_file=test_list,
    )

    return train_raw, test_norm, mean, std


# ======================== Helper Utilities ========================
def _find_image_dir(data_dir):
    """Auto-detect the images subfolder."""
    for candidate in ["images_224", "images-224/images-224", "images-224", "images", "."]:
        path = os.path.join(data_dir, candidate)
        if os.path.isdir(path):
            # Check if it contains png files
            if any(f.endswith(".png") for f in os.listdir(path)[:10]):
                return path
    raise FileNotFoundError(
        f"Cannot find image folder in {data_dir}. "
        f"Expected a subfolder named 'images_224' or 'images' containing .png files."
    )


def _find_csv(data_dir):
    """Auto-detect the metadata CSV."""
    for candidate in [
        "Data_Entry_2017_v2020.csv",
        "Data_Entry_2017.csv",
    ]:
        path = os.path.join(data_dir, candidate)
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(
        f"Cannot find Data_Entry CSV in {data_dir}."
    )


def _find_file(data_dir, name):
    """Find a file, return None if missing."""
    path = os.path.join(data_dir, name)
    return path if os.path.isfile(path) else None


def build_class_index(ds: Dataset, num_classes: int = NUM_CLASSES) -> Dict[int, List[int]]:
    """
    Build a mapping: class_id -> list of sample indices in the dataset.
    """
    class_index = {c: [] for c in range(num_classes)}
    targets = ds.targets if hasattr(ds, "targets") else [ds[i][1] for i in range(len(ds))]
    for i, y in enumerate(targets):
        class_index[int(y)].append(i)
    return class_index


def sample_images_per_class(
    ds: Dataset,
    class_index: Dict[int, List[int]],
    ipc: int,
    rng: random.Random,
    num_classes: int = NUM_CLASSES,
) -> torch.Tensor:
    """
    Sample IPC images per class from the dataset.

    Returns:
        images: [num_classes, IPC, 1, H, W] (raw, in [0,1])
    """
    per_class_imgs = []
    for c in range(num_classes):
        idxs = rng.sample(class_index[c], min(ipc, len(class_index[c])))
        imgs = [ds[i][0] for i in idxs]
        per_class_imgs.append(torch.stack(imgs, dim=0))    # [IPC, 1, H, W]
    return torch.stack(per_class_imgs, dim=0)               # [C, IPC, 1, H, W]


def make_labels_grid(ipc: int, num_classes: int = NUM_CLASSES) -> torch.Tensor:
    """
    Create aligned label grid: shape [num_classes, IPC].
    """
    return torch.arange(num_classes, dtype=torch.long).view(num_classes, 1).repeat(1, ipc)
