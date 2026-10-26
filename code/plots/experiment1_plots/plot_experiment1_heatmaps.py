"""
plot_experiment1_heatmaps.py
----------------------------
Processes the raw JSON outputs of Experiment 1 (Cross-Detection and Model Scale)
and generates sorted heatmaps for AUROC and PR-AUC.

This script scans the results directory for completed `run.py` executions,
extracts the detector and generator configurations alongside their evaluation
metrics, and saves the aggregated matrices as CSV files. It then uses these
matrices to plot and save high-resolution heatmaps, sorting the models from
largest to smallest parameter count to visually highlight the inverse
relationship between model scale and detection efficacy.
"""

import os
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Root directory containing the raw JSON outputs for Experiment 1
RESULTS_ROOT = "../../results/experiment1"

# Hardcoded sorting order for the heatmap axes.
# Models are ordered from largest parameter count (top/left) to smallest (bottom/right)
# to make the scale-efficacy gradient visually apparent.
LARGEST_TO_SMALLEST = [
    'HuggingFaceTB/SmolLM2-1.7B',
    'gpt2-xl',                     
    'EleutherAI/pythia-1.4b',    
    'EleutherAI/gpt-neo-1.3B',     
    'facebook/opt-1.3b',          
    'gpt2-large',                  
    'EleutherAI/pythia-410m',      
    'HuggingFaceTB/SmolLM2-360M',  
    'gpt2-medium',                
    'facebook/opt-350m',          
    'EleutherAI/pythia-160m',     
    'HuggingFaceTB/SmolLM2-135M',  
    'EleutherAI/gpt-neo-125M',    
    'facebook/opt-125m',           
    'gpt2',                     
    'distilgpt2',                
    'EleutherAI/pythia-70m'       
]

def extract_and_save_csvs():
    """
    Scans the results directory, extracts metrics from JSON files, 
    and saves the aggregated pivot matrices to CSV.
    
    Handles both the refactored argument names (generator_name, detector_name)
    and the legacy argument names (base_model_name, scoring_model_name) to 
    ensure backwards compatibility with older experiment runs.
    
    Returns:
        bool: True if data was successfully extracted and saved, False otherwise.
    """
    aggregated_data = []
    print("Scanning directories for 'cross_' experiment results...")

    for root, dirs, files in os.walk(RESULTS_ROOT):
        # Normalize path separators and ensure we are inside a 'cross_' experiment folder
        normalized_root = root.replace('\\', '/')
        if "cross_" not in normalized_root:
            continue  

        if "args.json" in files and "perturbation_50_d_results.json" in files:
            args_path = os.path.join(root, "args.json")
            results_path = os.path.join(root, "perturbation_50_d_results.json")
            
            try:
                with open(args_path, "r") as f:
                    args_data = json.load(f)
                    # Fallback to legacy argument names for older runs
                    gen_model = args_data.get("generator_name") or args_data.get("base_model_name")
                    det_model = args_data.get("detector_name") or args_data.get("scoring_model_name")
                    dataset = args_data.get("dataset")
                
                with open(results_path, "r") as f:
                    results_data = json.load(f)
                    roc_auc = results_data.get("metrics", {}).get("roc_auc")
                    pr_auc = results_data.get("pr_metrics", {}).get("pr_auc")
                
                if roc_auc is not None or pr_auc is not None:
                    aggregated_data.append({
                        "Generator Model": gen_model,
                        "Detector Model": det_model,
                        "Dataset": dataset,
                        "ROC-AUC": float(roc_auc) if roc_auc is not None else None,
                        "PR-AUC": float(pr_auc) if pr_auc is not None else None
                    })
                    
            except Exception as e:
                print(f"Error parsing folder {root}: {e}")

    df = pd.DataFrame(aggregated_data)

    if df.empty:
        print("No valid 'cross_' results found. Double-check your RESULTS_ROOT path.")
        return False
    
    print(f"\nSuccessfully collected {len(df)} 'cross_' experiment runs!")
    
    # Transform and save ROC-AUC matrix
    if "ROC-AUC" in df.columns and not df["ROC-AUC"].isna().all():
        roc_matrix = df.pivot_table(
            index="Generator Model", 
            columns="Detector Model", 
            values="ROC-AUC", 
            aggfunc="mean" 
        )
        roc_matrix.to_csv("cross_heatmap_roc_auc_matrix.csv")
        print("Saved matrix layout to 'cross_heatmap_roc_auc_matrix.csv'")

    # Transform and save PR-AUC matrix
    if "PR-AUC" in df.columns and not df["PR-AUC"].isna().all():
        pr_matrix = df.pivot_table(
            index="Generator Model", 
            columns="Detector Model", 
            values="PR-AUC", 
            aggfunc="mean" 
        )
        pr_matrix.to_csv("cross_heatmap_pr_auc_matrix.csv")
        print("Saved matrix layout to 'cross_heatmap_pr_auc_matrix.csv'")
        
    return True


def generate_heatmap(metric):
    """
    Generates and saves a high-resolution heatmap image for the specified metric.
    
    Args:
        metric (str): The metric to plot, either 'roc_auc' or 'pr_auc'.
    """
    csv_filename = f'cross_heatmap_{metric}_matrix.csv'
    png_filename = f'cross_heatmap_{metric}_matrix.png'
    
    display_metric = "AUROC" if metric == 'roc_auc' else "PR-AUC"

    if not os.path.exists(csv_filename):
        print(f"\nFile {csv_filename} not found. Skipping heatmap generation for {display_metric}...")
        return

    print(f"\nGenerating heatmap for {display_metric}...")
    
    df = pd.read_csv(csv_filename, index_col=0)

    # Drop models (rows/columns) that are completely blank (NaN)
    df_clean = df.dropna(how='all', axis=0).dropna(how='all', axis=1)

    # Filter the hardcoded sorting list to only include models actually present in the data
    valid_generators = [m for m in LARGEST_TO_SMALLEST if m in df_clean.index]
    valid_detectors = [m for m in LARGEST_TO_SMALLEST if m in df_clean.columns]

    # Reindex rows and columns to enforce the largest-to-smallest sorting
    df_sorted = df_clean.reindex(index=valid_generators, columns=valid_detectors)

    plt.figure(figsize=(16, 10))
    sns.heatmap(df_sorted, annot=True, cmap="mako", vmin=0.0, vmax=1.0, fmt=".2f")

    plt.title(f"{display_metric} of Distinguishing Human Text from Various Generator Models", pad=20, fontsize=14)
    plt.ylabel("Generative Model (Largest to Smallest)", labelpad=10, fontsize=12)
    plt.xlabel("Detector Model (Largest to Smallest)", labelpad=10, fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)

    plt.tight_layout()
    plt.savefig(png_filename, dpi=300)
    print(f"Saved heatmap image to {png_filename}")
    plt.close() 


def main():
    """Orchestrates the extraction of JSON results and the generation of heatmaps."""
    success = extract_and_save_csvs()
    
    if success:
        generate_heatmap('roc_auc')
        generate_heatmap('pr_auc')
        print("\nAll tasks completed successfully!")

if __name__ == '__main__':
    main()