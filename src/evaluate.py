"""
Evaluate a trained checkpoint on a split (default: test) and report the metrics from the plan:
accuracy, per-class precision / recall / F1 + macro, confusion matrix, ROC AUC,
overall and separately per dataset (SIPaKMeD, Herlev). Also measures inference time per image.

Examples (from the project root):
  python src/evaluate.py --run results/resnet50_sipakmed-herlev_seed42
  Cross-dataset (model trained only on SIPaKMeD, tested on the whole of Herlev):
  python src/evaluate.py --run results/resnet50_sipakmed_seed42 --split all --datasets herlev

Output: <run>/eval_<split>_<datasets>/{metrics.json, predictions.csv, confusion_*.png, roc.png}
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             roc_auc_score, roc_curve)
from torch.utils.data import DataLoader

from dataset import CervicalCellDataset, get_transforms
from models import build_model
from train import get_device

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "xpu":
        torch.xpu.synchronize()


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    probs, forward_time = [], 0.0
    for x, _ in loader:
        x = x.to(device)
        sync(device)
        t0 = time.perf_counter()
        logits = model(x)
        sync(device)
        forward_time += time.perf_counter() - t0
        probs.append(torch.softmax(logits, dim=1).cpu())
    probs = torch.cat(probs).numpy()
    return probs, forward_time / len(probs)


def compute_metrics(y_true, probs, classes):
    y_pred = probs.argmax(axis=1)
    report = classification_report(y_true, y_pred, labels=list(range(len(classes))),
                                   target_names=classes, output_dict=True, zero_division=0)
    metrics = {
        "n": int(len(y_true)),
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_precision": report["macro avg"]["precision"],
        "macro_recall": report["macro avg"]["recall"],
        "macro_f1": report["macro avg"]["f1-score"],
        "per_class": {c: {k: report[c][k] for k in ("precision", "recall", "f1-score", "support")}
                      for c in classes},
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=list(range(len(classes)))).tolist(),
    }
    if len(np.unique(y_true)) == len(classes):
        if len(classes) == 2:
            metrics["roc_auc"] = roc_auc_score(y_true, probs[:, 1])
        else:
            metrics["roc_auc"] = roc_auc_score(y_true, probs, multi_class="ovr")
    return metrics


def plot_confusion(cm, classes, title, path):
    cm = np.array(cm)
    fig, ax = plt.subplots(figsize=(4, 3.5))
    ax.imshow(cm, cmap="Blues")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xticks(range(len(classes)), classes)
    ax.set_yticks(range(len(classes)), classes)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_roc(curves, path):
    fig, ax = plt.subplots(figsize=(4.5, 4))
    for name, (fpr, tpr, auc) in curves.items():
        ax.plot(fpr, tpr, label=f"{name} (AUC {auc:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate (recall abnormal)")
    ax.set_title("ROC")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True, help="results/<run> folder with best.pt")
    parser.add_argument("--split", default="test", choices=["test", "val", "train", "all"])
    parser.add_argument("--datasets", nargs="+", default=["sipakmed", "herlev"])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    device = get_device()
    ckpt = torch.load(args.run / "best.pt", map_location="cpu", weights_only=False)
    train_args, classes = ckpt["args"], ckpt["classes"]
    label_col = "label" if classes == ["normal", "abnormal"] else "class_name"

    model = build_model(train_args["model"], len(classes), pretrained=False)
    model.load_state_dict(ckpt["model"])
    model.to(device)

    ds = CervicalCellDataset(args.split, get_transforms(False), args.datasets, label_col)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    print(f"Device: {device} | model: {train_args['model']} (epoch {ckpt['epoch']}) | "
          f"split: {args.split} | datasets: {args.datasets} | {len(ds)} images")

    probs, sec_per_image = predict(model, loader, device)
    y_true = np.array(ds.targets)

    out_dir = args.run / f"eval_{args.split}_{'-'.join(args.datasets)}"
    out_dir.mkdir(exist_ok=True)

    pred_df = ds.df[["path", "dataset", "class_name"]].copy()
    pred_df["true"] = y_true
    pred_df["pred"] = probs.argmax(axis=1)
    for i, c in enumerate(classes):
        pred_df[f"prob_{c}"] = probs[:, i]
    pred_df.to_csv(out_dir / "predictions.csv", index=False)

    # Overall + per dataset
    results = {"run": str(args.run), "model": train_args["model"], "trained_on": train_args["datasets"],
               "split": args.split, "inference_ms_per_image": sec_per_image * 1000, "subsets": {}}
    subsets = {"all": np.ones(len(ds), dtype=bool)}
    if len(args.datasets) > 1:
        for name in args.datasets:
            subsets[name] = (ds.df["dataset"] == name).to_numpy()

    curves = {}
    for name, mask in subsets.items():
        m = compute_metrics(y_true[mask], probs[mask], classes)
        results["subsets"][name] = m
        line = (f"{name:9s} n={m['n']:4d} | acc {m['accuracy']:.4f} | macro F1 {m['macro_f1']:.4f}"
                f" | AUC {m.get('roc_auc', float('nan')):.4f}")
        if label_col == "label":
            line += f" | recall abnormal {m['per_class']['abnormal']['recall']:.4f}"
        print(line)
        if plt is not None:
            plot_confusion(m["confusion_matrix"], classes, f"{train_args['model']} - {name}",
                           out_dir / f"confusion_{name}.png")
            if len(classes) == 2 and "roc_auc" in m:
                fpr, tpr, _ = roc_curve(y_true[mask], probs[mask][:, 1])
                curves[name] = (fpr, tpr, m["roc_auc"])

    if plt is not None and curves:
        plot_roc(curves, out_dir / "roc.png")
    elif plt is None:
        print("matplotlib not installed: skipping plots (pip install matplotlib)")

    (out_dir / "metrics.json").write_text(json.dumps(results, indent=2))
    print(f"Inference: {sec_per_image * 1000:.2f} ms / image (forward only, batch {args.batch_size})")
    print(f"Saved to {out_dir}")


if __name__ == "__main__":
    main()
