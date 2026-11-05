"""
Aggregate all evaluated runs into a comparison table: mean ± std over seeds, per model.

Reads results/<model>_<datasets>_seed<seed>/summary.json and its eval_<split>_<datasets>/metrics.json
and predictions.csv. Only runs that have the requested evaluation are included.

Run from the project root:  python src/compare.py
Output: results/comparison.csv (one row per run), results/comparison.md (mean ± std table),
        results/accuracy_by_class.csv (per original class, mean over seeds)
"""
import argparse
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

METRICS = [  # (column name, label in the table, scale)
    ("acc", "Accuracy (%)", 100),
    ("macro_f1", "Macro F1", 1),
    ("auc", "ROC AUC", 1),
    ("recall_abnormal", "Recall abnormal (%)", 100),
    ("recall_normal", "Recall normal (%)", 100),
    ("precision_abnormal", "Precision abnormal (%)", 100),
]
EXTRA = [
    ("best_epoch", "Best epoch", 1),
    ("train_minutes", "Training time (min)", 1),
    ("inference_ms", "Inference (ms / image)", 1),
]


def load_runs(eval_name: str) -> pd.DataFrame:
    rows = []
    for summary_path in sorted(RESULTS.glob("*/summary.json")):
        run_dir = summary_path.parent
        metrics_path = run_dir / eval_name / "metrics.json"
        if not metrics_path.exists():
            continue
        summary = json.loads(summary_path.read_text())
        metrics = json.loads(metrics_path.read_text())
        model, rest = run_dir.name.split("_", 1)
        trained_on, seed = rest.rsplit("_seed", 1)
        for subset, m in metrics["subsets"].items():
            pc = m["per_class"]
            rows.append({
                "run": run_dir.name, "model": model, "trained_on": trained_on, "seed": int(seed),
                "subset": subset, "n": m["n"],
                "acc": m["accuracy"], "macro_f1": m["macro_f1"], "auc": m.get("roc_auc"),
                "recall_abnormal": pc["abnormal"]["recall"], "recall_normal": pc["normal"]["recall"],
                "precision_abnormal": pc["abnormal"]["precision"],
                "best_epoch": summary["best_epoch"], "train_minutes": summary["train_minutes"],
                "inference_ms": metrics["inference_ms_per_image"], "parameters": summary["parameters"],
            })
    return pd.DataFrame(rows)


def mean_std(values: pd.Series, scale: float, digits: int) -> str:
    v = values.dropna() * scale
    if len(v) == 0:
        return "-"
    if len(v) == 1:
        return f"{v.iloc[0]:.{digits}f}"
    return f"{v.mean():.{digits}f} ± {v.std(ddof=1):.{digits}f}"


def markdown_table(df: pd.DataFrame) -> str:
    models = list(df["model"].unique())
    lines = []
    for subset in df["subset"].unique():
        part = df[df["subset"] == subset]
        n_seeds = part.groupby("model")["seed"].nunique().to_dict()
        lines.append(f"\n### Test subset: {subset} (n = {part['n'].iloc[0]})\n")
        lines.append("| Metric | " + " | ".join(f"{m} ({n_seeds.get(m, 0)} seeds)" for m in models) + " |")
        lines.append("| --- |" + " --- |" * len(models))
        rows = METRICS + (EXTRA if subset == "all" else [])
        for col, label, scale in rows:
            digits = 2 if scale == 100 or col in ("train_minutes", "inference_ms") else 4
            if col == "best_epoch":
                digits = 1
            cells = [mean_std(part[part["model"] == m][col], scale, digits) for m in models]
            lines.append(f"| {label} | " + " | ".join(cells) + " |")
        if subset == "all":
            params = [f"{part[part['model'] == m]['parameters'].iloc[0] / 1e6:.1f} M" for m in models]
            lines.append("| Parameters | " + " | ".join(params) + " |")
    return "\n".join(lines)


def accuracy_by_class(df: pd.DataFrame, eval_name: str) -> pd.DataFrame:
    parts = []
    for run in df["run"].unique():
        p = pd.read_csv(RESULTS / run / eval_name / "predictions.csv")
        p["correct"] = p["true"] == p["pred"]
        g = p.groupby(["dataset", "class_name"])["correct"].agg(["sum", "count"]).reset_index()
        g["run"] = run
        g["model"] = run.split("_", 1)[0]
        parts.append(g)
    all_runs = pd.concat(parts)
    all_runs["acc"] = all_runs["sum"] / all_runs["count"]
    table = all_runs.pivot_table(index=["dataset", "class_name", "count"], columns="model",
                                 values="acc", aggfunc="mean").reset_index()
    return table.rename(columns={"count": "n_test"})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", default="eval_test_sipakmed-herlev", help="evaluation folder name")
    parser.add_argument("--trained-on", default="sipakmed-herlev")
    args = parser.parse_args()

    df = load_runs(args.eval)
    df = df[df["trained_on"] == args.trained_on]
    if df.empty:
        raise SystemExit(f"No runs with {args.eval}/metrics.json trained on {args.trained_on}")

    df.to_csv(RESULTS / "comparison.csv", index=False)
    md = markdown_table(df)
    (RESULTS / "comparison.md").write_text(f"# Comparison ({args.eval})\n{md}\n", encoding="utf-8")
    print(md)

    by_class = accuracy_by_class(df, args.eval)
    by_class.to_csv(RESULTS / "accuracy_by_class.csv", index=False)
    print("\nAccuracy per original class (mean over seeds):")
    print(by_class.round(3).to_string(index=False))
    print(f"\nSaved: {RESULTS / 'comparison.csv'}, comparison.md, accuracy_by_class.csv")


if __name__ == "__main__":
    main()
