"""
Dataset and DataLoaders built from the CSVs in splits/ (created by make_splits.py).

The CSVs are the single source of truth for the split: nothing is re-split here.
"""
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2

ROOT = Path(__file__).resolve().parent.parent
SPLITS = ROOT / "splits"

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_transforms(train: bool, size: int = 224):
    """Resize + ImageNet normalization for every split; augmentation only for train."""
    base = [
        v2.ToImage(),
        v2.Resize((size, size), antialias=True),
    ]
    if train:
        aug = [
            v2.RandomHorizontalFlip(),
            v2.RandomVerticalFlip(),
            # 90° rotations leave no empty corners; cells have no natural orientation
            v2.RandomChoice([v2.RandomRotation((a, a)) for a in (0, 90, 180, 270)]),
            v2.RandomAffine(degrees=15, translate=(0.05, 0.05), scale=(0.9, 1.1)),
            v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15, hue=0.02),
            v2.RandomApply([v2.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))], p=0.2),
        ]
    else:
        aug = []
    tail = [
        v2.ToDtype(torch.float32, scale=True),
        *([v2.RandomApply([v2.GaussianNoise(sigma=0.02)], p=0.2)] if train else []),
        v2.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
    return v2.Compose(base + aug + tail)


class CervicalCellDataset(Dataset):
    """
    One row of the CSV = one cropped cell.

    label_col="label"      -> binary task (0 = normal, 1 = abnormal)
    label_col="class_name" -> multi-class task (classes sorted alphabetically)
    """

    def __init__(self, split: str, transform=None, datasets=None, label_col: str = "label"):
        df = pd.read_csv(SPLITS / f"{split}.csv")
        if datasets is not None:
            df = df[df["dataset"].isin(datasets)]
        if df.empty:
            raise ValueError(f"No rows for split={split}, datasets={datasets}")
        self.df = df.reset_index(drop=True)
        self.transform = transform

        if label_col == "label":
            self.classes = ["normal", "abnormal"]
            self.targets = self.df["label"].astype(int).tolist()
        else:
            self.classes = sorted(self.df[label_col].unique())
            index = {c: i for i, c in enumerate(self.classes)}
            self.targets = [index[c] for c in self.df[label_col]]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        img = Image.open(ROOT / self.df.at[i, "path"]).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, self.targets[i]

    def class_weights(self) -> torch.Tensor:
        """Inverse-frequency weights: n_total / (n_classes * n_c)."""
        counts = torch.bincount(torch.tensor(self.targets), minlength=len(self.classes)).float()
        return counts.sum() / (len(self.classes) * counts)


def make_loaders(batch_size=32, datasets=None, label_col="label", num_workers=4, size=224):
    train_ds = CervicalCellDataset("train", get_transforms(True, size), datasets, label_col)
    val_ds = CervicalCellDataset("val", get_transforms(False, size), datasets, label_col)
    common = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        pin_memory=torch.cuda.is_available(),
    )
    train_loader = DataLoader(train_ds, shuffle=True, drop_last=True, **common)
    val_loader = DataLoader(val_ds, shuffle=False, **common)
    return train_loader, val_loader


if __name__ == "__main__":
    # Quick check: python src/dataset.py
    ds = CervicalCellDataset("train", get_transforms(True))
    x, y = ds[0]
    print(f"train: {len(ds)} images, sample shape {tuple(x.shape)}, label {y}")
    print(f"class weights: {ds.class_weights().tolist()}")
