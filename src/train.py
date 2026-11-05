"""
Two-phase fine-tuning of ResNet50 / DenseNet121 on the cervical cell splits.

Phase 1: only the new head is trained (backbone frozen).
Phase 2: the whole model is trained with a lower LR, ReduceLROnPlateau and early stopping.
The best checkpoint (lowest validation loss over both phases) is saved.

Examples (from the project root):
  Short sanity run:  python src/train.py --model resnet50 --epochs1 1 --epochs2 1 --max-batches 20
  Full run:          python src/train.py --model resnet50 --seed 42
"""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from dataset import make_loaders
from models import MODEL_NAMES, build_model, count_parameters, get_head, set_backbone_trainable

ROOT = Path(__file__).resolve().parent.parent


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    return torch.device("cpu")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        torch.xpu.manual_seed_all(seed)


def run_epoch(model, loader, criterion, device, optimizer=None, max_batches=None):
    """One pass over the loader. Trains if an optimizer is given, otherwise only evaluates."""
    training = optimizer is not None
    model.train(training)
    total_loss, correct, n = 0.0, 0, 0
    tp, fn = 0, 0  # for recall of the abnormal class (label 1)

    with torch.set_grad_enabled(training):
        for b, (x, y) in enumerate(loader):
            if max_batches is not None and b >= max_batches:
                break
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            logits = model(x)
            loss = criterion(logits, y)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            pred = logits.argmax(dim=1)
            total_loss += loss.item() * y.size(0)
            correct += (pred == y).sum().item()
            n += y.size(0)
            tp += ((pred == 1) & (y == 1)).sum().item()
            fn += ((pred != 1) & (y == 1)).sum().item()

    return {
        "loss": total_loss / max(n, 1),
        "acc": correct / max(n, 1),
        "recall_abn": tp / max(tp + fn, 1),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODEL_NAMES, default="resnet50")
    parser.add_argument("--datasets", nargs="+", default=["sipakmed", "herlev"],
                        help="which datasets to train/validate on (cross-dataset: pass only one)")
    parser.add_argument("--epochs1", type=int, default=5, help="phase 1 epochs (head only)")
    parser.add_argument("--epochs2", type=int, default=25, help="phase 2 max epochs (full model)")
    parser.add_argument("--lr1", type=float, default=1e-3)
    parser.add_argument("--lr2", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--patience", type=int, default=6, help="early stopping patience (phase 2)")
    parser.add_argument("--no-class-weights", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-batches", type=int, default=None, help="limit batches per epoch (sanity run)")
    parser.add_argument("--no-pretrained", action="store_true", help="random init (testing only)")
    parser.add_argument("--out", type=Path, default=ROOT / "results")
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    run_name = f"{args.model}_{'-'.join(args.datasets)}_seed{args.seed}"
    out_dir = args.out / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Device: {device} | run: {run_name}")

    train_loader, val_loader = make_loaders(args.batch_size, args.datasets, num_workers=args.num_workers)
    train_ds = train_loader.dataset
    print(f"Train: {len(train_ds)} | Val: {len(val_loader.dataset)} | classes: {train_ds.classes}")

    model = build_model(args.model, len(train_ds.classes), pretrained=not args.no_pretrained).to(device)
    print(f"Parameters: {count_parameters(model) / 1e6:.1f} M")

    weights = None if args.no_class_weights else train_ds.class_weights().to(device)
    if weights is not None:
        print(f"Class weights: {[round(w, 3) for w in weights.tolist()]}")
    criterion = nn.CrossEntropyLoss(weight=weights)

    history, best_val, best_epoch, epoch = [], float("inf"), -1, 0
    ckpt_path = out_dir / "best.pt"
    start = time.time()

    def log_and_checkpoint(phase, tr, va, lr):
        nonlocal best_val, best_epoch
        improved = va["loss"] < best_val
        if improved:
            best_val, best_epoch = va["loss"], epoch
            torch.save({"model": model.state_dict(), "args": vars(args) | {"out": str(args.out)},
                        "classes": train_ds.classes, "epoch": epoch}, ckpt_path)
        history.append({"epoch": epoch, "phase": phase, "lr": lr,
                        **{f"train_{k}": v for k, v in tr.items()},
                        **{f"val_{k}": v for k, v in va.items()}})
        print(f"[{phase}] epoch {epoch:3d} | lr {lr:.1e} | "
              f"train loss {tr['loss']:.4f} acc {tr['acc']:.3f} | "
              f"val loss {va['loss']:.4f} acc {va['acc']:.3f} recall_abn {va['recall_abn']:.3f}"
              f"{'  *' if improved else ''}")
        return improved

    # Phase 1: head only
    set_backbone_trainable(model, False)
    optimizer = torch.optim.AdamW(get_head(model).parameters(), lr=args.lr1, weight_decay=args.weight_decay)
    for _ in range(args.epochs1):
        epoch += 1
        tr = run_epoch(model, train_loader, criterion, device, optimizer, args.max_batches)
        va = run_epoch(model, val_loader, criterion, device, max_batches=args.max_batches)
        log_and_checkpoint("phase1", tr, va, args.lr1)

    # Phase 2: full model
    set_backbone_trainable(model, True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr2, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=3)
    bad_epochs = 0
    for _ in range(args.epochs2):
        epoch += 1
        tr = run_epoch(model, train_loader, criterion, device, optimizer, args.max_batches)
        va = run_epoch(model, val_loader, criterion, device, max_batches=args.max_batches)
        improved = log_and_checkpoint("phase2", tr, va, optimizer.param_groups[0]["lr"])
        scheduler.step(va["loss"])
        bad_epochs = 0 if improved else bad_epochs + 1
        if bad_epochs >= args.patience:
            print(f"Early stopping: no val improvement for {args.patience} epochs")
            break

    minutes = (time.time() - start) / 60
    pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False)
    summary = {"run": run_name, "device": str(device), "best_epoch": best_epoch,
               "best_val_loss": best_val, "epochs_run": epoch, "train_minutes": round(minutes, 2),
               "parameters": count_parameters(model)}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nDone in {minutes:.1f} min. Best epoch {best_epoch} (val loss {best_val:.4f})")
    print(f"Saved: {ckpt_path}, history.csv, summary.json")


if __name__ == "__main__":
    main()
