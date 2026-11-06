"""
Grad-CAM and Grad-CAM++ with the pytorch-grad-cam library (pip install grad-cam).

Same cells, same figure layout and same options as gradcam.py (the hand-made version, kept as a fallback):
cells are picked from the FIRST run's predictions.csv in four groups (tp / tn / fp / fn), and each figure shows
the cell, then Grad-CAM and Grad-CAM++ for every run. The heatmap explains the class each model predicted.

Examples (from the project root):
  python src/gradcam_lib.py --runs results/resnet50_sipakmed-herlev_seed42 results/densenet121_sipakmed-herlev_seed42
  python src/gradcam_lib.py --runs results/resnet50_sipakmed-herlev_seed42 results/densenet121_sipakmed-herlev_seed42 \
      --classes normal_columnar --groups fp tn

Output: results/gradcam_lib/<group>[_<classes>].png  (+ gradcam_cells.csv)
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dataset import get_transforms
from models import build_model
from train import get_device

ROOT = Path(__file__).resolve().parent.parent

GROUP_TITLES = {
    "tp": "Abnormal cells correctly detected (TP)",
    "tn": "Normal cells correctly recognised (TN)",
    "fp": "False alarms: normal cells predicted abnormal (FP)",
    "fn": "Missed abnormal cells (FN)",
}

def target_layer(model, name):
    """Last convolutional block: its feature maps still keep the spatial layout (7 × 7)."""
    if name == "resnet50":
        return model.layer4[-1]
    if name == "densenet121":
        return model.features.norm5
    raise ValueError(name)

class Clone(nn.Module):
    """DenseNet applies an in-place ReLU right after `features`; the library keeps a reference to the
    activation of the target layer, so we hand the in-place ReLU a copy instead."""

    def forward(self, x):
        return x.clone()


class LibCAM:
    def __init__(self, run_dir: Path, device):
        ckpt = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)
        self.name = ckpt["args"]["model"]
        self.classes = ckpt["classes"]
        self.model = build_model(self.name, len(self.classes), pretrained=False)
        self.model.load_state_dict(ckpt["model"])
        if self.name == "densenet121":
            self.model.features.add_module("clone_for_cam", Clone())
        self.model.to(device).eval()
        self.label = f"{self.name} (seed {ckpt['args']['seed']})"
        layers = [target_layer(self.model, self.name)]
        self.gradcam = GradCAM(model=self.model, target_layers=layers)
        self.gradcam_pp = GradCAMPlusPlus(model=self.model, target_layers=layers)
        self.device = device

    def __call__(self, x):
        """x: 1 × 3 × H × W. Returns (predicted class, its probability, grad-cam, grad-cam++), maps H × W in [0, 1]."""
        with torch.no_grad():
            probs = torch.softmax(self.model(x.to(self.device)), dim=1)[0].cpu()
        cls = int(probs.argmax())
        targets = [ClassifierOutputTarget(cls)]
        heat = self.gradcam(input_tensor=x, targets=targets)[0]
        heat_pp = self.gradcam_pp(input_tensor=x, targets=targets)[0]
        return cls, float(probs[cls]), heat, heat_pp

def pick_cells(pred_csv: Path, groups, classes, n, seed):
    p = pd.read_csv(pred_csv)
    if classes:
        p = p[p["class_name"].isin(classes)]
    p["group"] = np.select(
        [(p.true == 1) & (p.pred == 1), (p.true == 0) & (p.pred == 0),
         (p.true == 0) & (p.pred == 1), (p.true == 1) & (p.pred == 0)],
        ["tp", "tn", "fp", "fn"], default="")
    picked = {}
    for gname in groups:
        part = p[p["group"] == gname]
        if part.empty:
            print(f"[{gname}] no cells")
            continue
        # spread the picks over the original classes, then fill up randomly
        part = part.sample(frac=1, random_state=seed)
        first = part.groupby("class_name", sort=False).head(1)
        rest = part.drop(first.index)
        picked[gname] = pd.concat([first, rest]).head(n)
    return picked

def draw(cells: pd.DataFrame, cams, title, path, size=224):
    to_tensor = get_transforms(False, size)
    n_cols = 1 + 2 * len(cams)
    fig, axes = plt.subplots(len(cells), n_cols, figsize=(2.3 * n_cols, 3.0 * len(cells)), squeeze=False)
    for r, (_, row) in enumerate(cells.iterrows()):
        img = Image.open(ROOT / row["path"]).convert("RGB")
        x = to_tensor(img)[None]
        rgb = np.asarray(img.resize((size, size)), dtype=np.float32) / 255
        truth = "abnormal" if row["true"] == 1 else "normal"
        axes[r, 0].imshow(rgb)
        axes[r, 0].set_title(f"{row['class_name']}\n({row['dataset']}, true: {truth})", fontsize=8.5)
        for k, cam in enumerate(cams):
            cls, prob, heat, heat_pp = cam(x)
            for j, (h, kind) in enumerate([(heat, "Grad-CAM"), (heat_pp, "Grad-CAM++")]):
                ax = axes[r, 1 + 2 * k + j]
                ax.imshow(show_cam_on_image(rgb, h, use_rgb=True, image_weight=0.55))
                color = "black" if cls == row["true"] else "red"
                ax.set_title(f"{cam.name} · {kind}\npred: {cam.classes[cls]} ({prob:.0%})", fontsize=8.5, color=color)
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97), h_pad=2.0)
    fig.savefig(path, dpi=130, facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, nargs="+", required=True, help="one or more results/<run> folders")
    parser.add_argument("--eval", default="eval_test_sipakmed-herlev",
                        help="evaluation folder of the FIRST run whose predictions.csv picks the cells")
    parser.add_argument("--groups", nargs="+", default=["tp", "tn", "fp", "fn"], choices=list(GROUP_TITLES))
    parser.add_argument("--classes", nargs="+", default=None, help="only these original classes")
    parser.add_argument("--n", type=int, default=6, help="cells per figure")
    parser.add_argument("--seed", type=int, default=0, help="which cells get picked")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "gradcam_lib")
    args = parser.parse_args()

    device = torch.device("cpu") if args.cpu else get_device()
    cams = [LibCAM(run, device) for run in args.runs]
    print(f"Device: {device} | models: {', '.join(c.label for c in cams)}")

    pred_csv = args.runs[0] / args.eval / "predictions.csv"
    if not pred_csv.exists():
        raise SystemExit(f"{pred_csv} not found: run evaluate.py on {args.runs[0]} first")
    picked = pick_cells(pred_csv, args.groups, args.classes, args.n, args.seed)

    args.out.mkdir(parents=True, exist_ok=True)
    suffix = f"_{'-'.join(args.classes)}" if args.classes else ""
    shown = []
    for gname, cells in picked.items():
        path = args.out / f"{gname}{suffix}.png"
        title = f"{GROUP_TITLES[gname]} · groups by {cams[0].label}" + (f" · {', '.join(args.classes)}" if args.classes else "")
        draw(cells, cams, title, path)
        shown.append(cells.assign(figure=path.name))
        print(f"[{gname}] {len(cells)} cells -> {path}")
    if shown:
        pd.concat(shown).to_csv(args.out / f"gradcam_cells{suffix}.csv", index=False)


if __name__ == "__main__":
    main()
