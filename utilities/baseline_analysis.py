import pandas as pd
import numpy as np
import logging
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix
)

def compute_baseline_vs_final(df, visit_cols, final_col="final_diagnosis (0-4)", output_folder=None):
    """
    Computes the baseline by comparing doctors' visit assessments with the
    final diagnosis.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset containing the visit columns and the final diagnosis
    visit_cols : list
        Visit columns to compare against the final diagnosis
    final_col : str
        Final diagnosis column
    output_folder : str, optional
        Folder where to save the baseline CSV

    Returns
    -------
    results : dict
        Dictionary with metrics for each visit_col
    """
    log = logging.info
    log("🔍 Starting baseline analysis")

    results = {}

    if final_col not in df.columns:
        raise ValueError(f"Column '{final_col}' not found in the dataset!")

    # We only consider complete rows for the relevant columns
    relevant_cols = [final_col] + visit_cols
    # df_clean = df[relevant_cols].dropna()

    # y_true = df_clean[final_col].astype(int)

    for col in visit_cols:
        df_clean = df[[final_col, col]].dropna()

        y_true = df_clean[final_col].astype(int)

        if col not in df_clean.columns:
            print(f"⚠️ Column '{col}' not found, skipping...")
            continue

        y_pred = df_clean[col].astype(int)

        # Global metrics
        accuracy = accuracy_score(y_true, y_pred)
        f1_weighted = f1_score(y_true, y_pred, average='weighted')
        precision_weighted = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall_weighted = recall_score(y_true, y_pred, average='weighted')

        # Per-class metrics
        report = pd.DataFrame({
            "precision": precision_score(y_true, y_pred, average=None, zero_division=0),
            "recall": recall_score(y_true, y_pred, average=None)
        }, index=np.unique(y_true))

        roc_auc = roc_auc_score(y_true, y_pred)

        # Assign per-class metrics
        metrics = {
            "accuracy": accuracy,
            "f1": f1_weighted,
            "precision": precision_weighted,
            "recall": recall_weighted,
            "auc": roc_auc
        }

        for cls in report.index:
            metrics[f"precision_{cls}"] = report.loc[cls, "precision"]
            metrics[f"recall_{cls}"] = report.loc[cls, "recall"]

        # Print to console
        print(f"\n🔹 Doctor baseline '{col}' vs '{final_col}':")
        log(f"\n🔹 Doctor baseline '{col}' vs '{final_col}':")
        for metric_name, value in metrics.items():
            print(f"  {metric_name}: {value:.4f}" if value is not None else f"  {metric_name}: None")

        results[col] = metrics

        # 🔹 Save confusion matrix
        if output_folder:
            cm = confusion_matrix(y_true, y_pred)
            fig, ax = plt.subplots(figsize=(6, 5))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            ax.set_title(f"Confusion Matrix - {col}")
            os.makedirs(output_folder, exist_ok=True)
            cm_path = os.path.join(output_folder, f"confusion_matrix_{col}.png")
            plt.tight_layout()
            plt.savefig(cm_path)
            plt.close(fig)
            print(f"💾 Confusion matrix saved to {cm_path}")
            log(f"Confusion matrix saved to {cm_path}")

    # Save CSV
    if output_folder:
        baseline_df = pd.DataFrame([
            {"visit": col, **metrics} for col, metrics in results.items()
        ])
        os.makedirs(output_folder, exist_ok=True)
        baseline_path = os.path.join(output_folder, "baseline_vs_final.csv")
        baseline_df.to_csv(baseline_path, index=False)
        print(f"\n💾 Baseline saved to {baseline_path}")
        log(f"\n💾 Baseline saved to {baseline_path}")

    return results
