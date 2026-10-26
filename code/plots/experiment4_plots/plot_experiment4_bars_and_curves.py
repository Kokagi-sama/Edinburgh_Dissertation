"""
plot_experiment4_bars_and_curves.py
-----------------------------------
Processes the raw JSON outputs of Experiment 4 (Linguistic Perturbation Strategies)
to generate grouped bar charts, degradation line plots, ROC/PR curves, and aggregate 
trend plots.

This script scans the results directory for completed `run.py` executions across
different perturbation strategies (Control, Synonyms, Random Words, Random Tokens),
extracts the evaluation metrics, and plots:
1. Grouped bar charts comparing AUROC and PR-AUC across strategies for each detector.
2. Line plots tracing the degradation of performance as linguistic structure is removed.
3. ROC and Precision-Recall curves for each detector, colored by perturbation strategy.
4. Aggregate trend plots showing the mean performance across all detectors (Figure 7.4).

The script infers the detector and treatment configurations directly from the 
directory structure, making it robust to changes in argument naming conventions.
"""

import os
import json
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Root directory containing the raw JSON outputs for Experiment 4
RESULTS_ROOT = "../../results/experiment4"

# Output directory for the generated figures
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures", "Experiment4")
os.makedirs(OUT_DIR, exist_ok=True)

# Maps directory names to clean display names for detectors and treatments
DET_MAP = {
    "pythia_1.4b": "Pythia-1.4B",
    "pythia_410m": "Pythia-410M",
    "pythia_70m": "Pythia-70M",
}
DET_ORDER = ["Pythia-1.4B", "Pythia-410M", "Pythia-70M"]
DET_COLORS = {"Pythia-1.4B": "#4C72B0", "Pythia-410M": "#DD8452", "Pythia-70M": "#55A868"}

TREAT_MAP = {
    "1_control_t5": "Control (T5)",
    "2_treatment_a_synonyms": "Synonyms",
    "3_treatment_b_random_words": "Random Words",
    "4_treatment_c_random_tokens": "Random Tokens",
}
TREAT_ORDER = ["Control (T5)", "Synonyms", "Random Words", "Random Tokens"]

# Global plot styling
plt.rcParams.update({"figure.dpi": 150, "axes.spines.top": False, "axes.spines.right": False})

def clean(obj):
    """Recursively strips stray whitespace from JSON keys and values."""
    if isinstance(obj, dict):
        return {str(k).strip(): clean(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [clean(v) for v in obj]
    if isinstance(obj, str):
        return obj.strip()
    return obj

def load_metrics(folder):
    """
    Reads the primary results JSON file. Falls back to merging chunked 
    JSON files if the primary file is missing.
    """
    full = sorted(glob.glob(os.path.join(folder, "**", "perturbation_50_d_results.json"), recursive=True))
    payloads = []
    
    if full:
        with open(full[0], encoding="utf-8") as fh:
            payloads.append(clean(json.load(fh)))
    else:
        merged = {}
        chunks = sorted(glob.glob(os.path.join(folder, "**", "perturbation_50_d_results_chunk_*.json"), recursive=True))
        for c in chunks:
            with open(c, encoding="utf-8") as fh:
                data = clean(json.load(fh))
            for key in ("metrics", "pr_metrics"):
                if key in data and key not in merged:
                    merged[key] = data[key]
        if merged:
            payloads.append(merged)

    if not payloads:
        return None
        
    p = payloads[0]
    m = p.get("metrics", {})
    pm = p.get("pr_metrics", {})
    
    if "roc_auc" not in m or "pr_auc" not in pm:
        return None
        
    return {
        "roc_auc": float(m["roc_auc"]),
        "pr_auc": float(pm["pr_auc"]),
        "fpr": m.get("fpr"),
        "tpr": m.get("tpr"),
        "precision": pm.get("precision"),
        "recall": pm.get("recall"),
    }

def extract_all():
    """
    Scans the results directory, extracts metrics from JSON files, and returns 
    a pandas DataFrame and a dictionary of curve data.
    """
    records, curves = [], {}
    
    if not os.path.isdir(RESULTS_ROOT):
        print(f"Results root not found: {RESULTS_ROOT}")
        return pd.DataFrame(records), curves

    for det_dir in sorted(os.listdir(RESULTS_ROOT)):
        det_path = os.path.join(RESULTS_ROOT, det_dir)
        if not os.path.isdir(det_path) or det_dir not in DET_MAP:
            continue
            
        det = DET_MAP[det_dir]
        
        for treat_dir in sorted(os.listdir(det_path)):
            treat_path = os.path.join(det_path, treat_dir)
            if not os.path.isdir(treat_path) or treat_dir not in TREAT_MAP:
                continue
                
            res = load_metrics(treat_path)
            if res is None:
                print(f"[warn] no valid results JSON under {det_dir}/{treat_dir}")
                continue
                
            records.append({
                "Detector": det, 
                "Treatment": TREAT_MAP[treat_dir],
                "roc_auc": res["roc_auc"], 
                "pr_auc": res["pr_auc"]
            })
            curves[(det, TREAT_MAP[treat_dir])] = res
            
    return pd.DataFrame(records), curves

def plot_bars(df):
    """Generates and saves grouped bar charts for AUROC and PR-AUC (Figure 7.1)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.6))
    x = np.arange(len(TREAT_ORDER))
    width = 0.25
    dets = [d for d in DET_ORDER if d in df["Detector"].unique()]
    
    for ax, metric, title in [(axes[0], "roc_auc", "AUROC"), (axes[1], "pr_auc", "PR-AUC")]:
        for i, det in enumerate(dets):
            vals = [df.loc[(df.Detector == det) & (df.Treatment == t), metric].mean() for t in TREAT_ORDER]
            bars = ax.bar(x + (i - 1) * width, vals, width, label=det, color=DET_COLORS[det])
            ax.bar_label(bars, fmt="%.3f", fontsize=7, padding=2)
            
        ax.axhline(0.5, ls="--", lw=1, color="grey", label="Chance Level (0.5)")
        ax.set_ylim(0, 1.08)
        ax.set_xticks(x)
        ax.set_xticklabels(TREAT_ORDER)
        ax.set_xlabel("Perturbation strategy")
        ax.set_ylabel(title)
        ax.set_title(f"{title} by perturbation strategy")
        ax.legend(fontsize=8)
        
    fig.tight_layout()
    out_path = os.path.join(OUT_DIR, "exp4_strategy_bars.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close(fig)

def plot_degradation(df):
    """Generates and saves line plots showing performance degradation (Figure 7.2)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))
    x = np.arange(len(TREAT_ORDER))
    dets = [d for d in DET_ORDER if d in df["Detector"].unique()]
    
    for ax, metric, title in [(axes[0], "roc_auc", "AUROC"), (axes[1], "pr_auc", "PR-AUC")]:
        for det in dets:
            vals = [df.loc[(df.Detector == det) & (df.Treatment == t), metric].mean() for t in TREAT_ORDER]
            ax.plot(x, vals, marker="o", lw=2, label=det, color=DET_COLORS[det])
            for xi, v in zip(x, vals):
                ax.annotate(f"{v:.2f}", (xi, v), textcoords="offset points",
                            xytext=(0, 6), ha="center", fontsize=7)
                            
        ax.axhline(0.5, ls="--", lw=1, color="grey", label="Chance Level (0.5)")
        ax.set_ylim(0, 1.08)
        ax.set_xticks(x)
        ax.set_xticklabels(TREAT_ORDER)
        ax.set_xlabel("Perturbation strategy (decreasing linguistic structure)")
        ax.set_ylabel(title)
        ax.set_title(f"{title} as linguistic structure is removed")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25)
        
    fig.tight_layout()
    out_path = os.path.join(OUT_DIR, "exp4_degradation_lines.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close(fig)

def plot_curves(curves):
    """Generates and saves ROC and Precision-Recall curves for each detector (Figure 7.3)."""
    dets = [d for d in DET_ORDER if any(d == k[0] for k in curves)]
    if not dets:
        return
        
    fig, axes = plt.subplots(2, len(dets), figsize=(5.2 * len(dets), 9))
    if len(dets) == 1:
        axes = np.array([axes])
        
    for col, det in enumerate(dets):
        ax = axes[0, col]
        for treat in TREAT_ORDER:
            c = curves.get((det, treat))
            if c and c.get("fpr"):
                ax.plot(c["fpr"], c["tpr"], lw=1.6,
                        label=f"{treat} ({c['roc_auc']:.3f})")
        ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="Chance Level")
        ax.set_title(f"ROC: {det}")
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
        ax.legend(fontsize=7)

        ax = axes[1, col]
        for treat in TREAT_ORDER:
            c = curves.get((det, treat))
            if c and c.get("recall"):
                ax.plot(c["recall"], c["precision"], lw=1.6,
                        label=f"{treat} ({c['pr_auc']:.3f})")
        ax.set_title(f"PR: {det}")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.legend(fontsize=7)
        
    fig.tight_layout()
    out_path = os.path.join(OUT_DIR, "exp4_roc_pr_curves.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close(fig)

def plot_trend(df):
    """
    Generates and saves the aggregate trend plot (Figure 7.4 in the thesis),
    showing mean AUROC and PR-AUC across all detectors for each perturbation strategy.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))
    x = np.arange(len(TREAT_ORDER))
    
    for ax, metric, title in [(axes[0], "roc_auc", "AUROC"), (axes[1], "pr_auc", "PR-AUC")]:
        means = [df[df.Treatment == t][metric].mean() for t in TREAT_ORDER]
        mins = [df[df.Treatment == t][metric].min() for t in TREAT_ORDER]
        maxs = [df[df.Treatment == t][metric].max() for t in TREAT_ORDER]
        
        ax.plot(x, means, marker="o", lw=2.5, color="black", label="Mean across detectors")
        ax.fill_between(x, mins, maxs, color="grey", alpha=0.2, label="Min-Max spread")
        
        for xi, v in zip(x, means):
            ax.annotate(f"{v:.2f}", (xi, v), textcoords="offset points",
                        xytext=(0, 8), ha="center", fontsize=9, fontweight="bold")
                        
        ax.axhline(0.5, ls="--", lw=1, color="grey", label="Chance Level (0.5)")
        ax.set_ylim(0, 1.08)
        ax.set_xticks(x)
        ax.set_xticklabels(TREAT_ORDER)
        ax.set_xlabel("Perturbation strategy")
        ax.set_ylabel(title)
        ax.set_title(f"Mean {title} by perturbation strategy")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25)
        
    fig.tight_layout()
    out_path = os.path.join(OUT_DIR, "exp4_strategy_trend.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close(fig)

def main():
    """Orchestrates the extraction of JSON results and the generation of all plots."""
    df, curves = extract_all()
    if df.empty:
        print("No Experiment 4 results found under", RESULTS_ROOT)
        return

    print(df.pivot(index="Detector", columns="Treatment", values=["roc_auc", "pr_auc"]).round(3))
    
    plot_bars(df)
    plot_degradation(df)
    plot_curves(curves)
    plot_trend(df)
    
    print("\nAll plotting tasks completed successfully!")

if __name__ == "__main__":
    main()