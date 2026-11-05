"""
Create the train / val / test split (70 / 15 / 15) for SIPaKMeD and Herlev.

- Unified labels: 0 = normal, 1 = abnormal (column `label`);
  the original class is kept in `class_name` (for the 5-class experiment).
- Each dataset is split separately, stratified on the original class and
  grouped by source image, so cells from the same source never end up in two splits.
  The per-dataset splits are then concatenated into train/val/test.
- The Kaggle version of Herlev ships with its own train/test folders; we ignore
  them and re-split, because they place cells from the same source in both.

Output: splits/all.csv, splits/train.csv, splits/val.csv, splits/test.csv
"""
import argparse
import re
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
SPLITS = ROOT / "splits"

SIPAKMED_CLASSES = {
    "Superficial-Intermediate": 0,
    "Parabasal": 0,
    "Metaplastic": 0,  # benign -> normal (project decision; change here if needed)
    "Koilocytotic": 1,
    "Dyskeratotic": 1,
}

HERLEV_CLASSES = {
    "normal_superficiel": 0,
    "normal_intermediate": 0,
    "normal_columnar": 0,
    "light_dysplastic": 1,
    "moderate_dysplastic": 1,
    "severe_dysplastic": 1,
    "carcinoma_in_situ": 1,
}

IMG_EXT = {".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def list_images(folder: Path):
    """All images under a folder, recursively, skipping masks (*-d.bmp) and kagglehub's .complete."""
    for p in sorted(folder.rglob("*")):
        if not p.is_file() or p.suffix.lower() not in IMG_EXT:
            continue
        if ".complete" in p.parts or p.stem.lower().endswith("-d"):
            continue
        yield p


def collect_sipakmed() -> pd.DataFrame:
    rows = []
    for p in list_images(DATA / "sipakmed"):
        if p.parent.name.upper() != "CROPPED":
            continue  # full multi-cell images are not used
        cls = p.parent.parent.name.removeprefix("im_")
        if cls not in SIPAKMED_CLASSES:
            print(f"[warning] unknown SIPaKMeD class: {cls} ({p})")
            continue
        # 001_01.bmp -> source image 001 (numbering restarts in every class folder)
        source = p.stem.split("_")[0]
        rows.append({
            "path": p.relative_to(ROOT).as_posix(),
            "dataset": "sipakmed",
            "class_name": cls,
            "label": SIPAKMED_CLASSES[cls],
            "group": f"sipakmed_{cls}_{source}",
        })
    return pd.DataFrame(rows)


def collect_herlev() -> pd.DataFrame:
    rows = []
    for p in list_images(DATA / "herlev"):
        cls = p.parent.name
        if cls not in HERLEV_CLASSES:
            print(f"[warning] unknown Herlev class: {cls} ({p})")
            continue
        # 149056321-149056343-001.BMP -> group on the first number
        m = re.match(r"^(\d+)-(\d+)-\d+$", p.stem)
        source = m.group(1) if m else p.stem
        rows.append({
            "path": p.relative_to(ROOT).as_posix(),
            "dataset": "herlev",
            "class_name": cls,
            "label": HERLEV_CLASSES[cls],
            "group": f"herlev_{source}",
            "file_name": p.name.lower(),
        })
    df = pd.DataFrame(rows)
    dup = df.duplicated(subset=["class_name", "file_name"])
    if dup.any():
        print(f"[warning] Herlev: {dup.sum()} duplicate files (same name in train and test), dropping them")
        df = df[~dup]
    return df.drop(columns="file_name")


def split_dataset(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    """20 stratified, grouped folds: 3 -> test (15%), 3 -> val (15%), 14 -> train (70%)."""
    df = df.reset_index(drop=True).copy()
    sgkf = StratifiedGroupKFold(n_splits=20, shuffle=True, random_state=seed)
    fold = pd.Series(-1, index=df.index)
    for k, (_, idx) in enumerate(sgkf.split(df, df["class_name"], df["group"])):
        fold.iloc[idx] = k
    df["split"] = "train"
    df.loc[fold < 3, "split"] = "test"
    df.loc[(fold >= 3) & (fold < 6), "split"] = "val"
    return df


def check(df: pd.DataFrame):
    leaks = df.groupby("group")["split"].nunique()
    assert (leaks == 1).all(), f"Groups found in more than one split: {list(leaks[leaks > 1].index)[:5]}"
    for (ds, sp), part in df.groupby(["dataset", "split"]):
        assert part["label"].nunique() == 2, f"{ds}/{sp} does not contain both classes"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    parts = []
    for name, collect in [("sipakmed", collect_sipakmed), ("herlev", collect_herlev)]:
        df = collect()
        if df.empty:
            raise SystemExit(f"No images found for {name} in {DATA / name}")
        print(f"{name}: {len(df)} cells, {df['group'].nunique()} source groups")
        parts.append(split_dataset(df, args.seed))

    all_df = pd.concat(parts, ignore_index=True)
    check(all_df)

    SPLITS.mkdir(exist_ok=True)
    all_df.to_csv(SPLITS / "all.csv", index=False)
    for sp in ["train", "val", "test"]:
        all_df[all_df["split"] == sp].to_csv(SPLITS / f"{sp}.csv", index=False)

    print("\nCells per dataset / split / label (0 = normal, 1 = abnormal):")
    print(pd.crosstab([all_df["dataset"], all_df["split"]], all_df["label"], margins=True))
    print("\nPercent per split:")
    print((all_df.groupby("dataset")["split"].value_counts(normalize=True) * 100).round(1))
    print(f"\nSaved to {SPLITS}")


if __name__ == "__main__":
    main()
