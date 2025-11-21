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
    Calcola baseline confrontando le visite dei dottori con la diagnosi definitiva.

    Parametri
    ----------
    df : pd.DataFrame
        Dataset contenente le colonne delle visite e la diagnosi definitiva
    visit_cols : list
        Colonne delle visite da confrontare con la diagnosi definitiva
    final_col : str
        Colonna della diagnosi definitiva
    output_folder : str, optional
        Cartella dove salvare il CSV con le baseline

    Ritorna
    -------
    results : dict
        Dizionario con metriche per ogni visit_col
    """
    log = logging.info
    log("🔍 Avvio analisi delle baseline")

    results = {}

    if final_col not in df.columns:
        raise ValueError(f"Colonna '{final_col}' non trovata nel dataset!")

    # Consideriamo solo righe complete per le colonne rilevanti
    relevant_cols = [final_col] + visit_cols
    df_clean = df[relevant_cols].dropna()

    y_true = df_clean[final_col].astype(int)

    for col in visit_cols:
        if col not in df_clean.columns:
            print(f"⚠️ Colonna '{col}' non trovata, salto...")
            continue

        y_pred = df_clean[col].astype(int)

        # Metriche globali
        accuracy = accuracy_score(y_true, y_pred)
        f1_weighted = f1_score(y_true, y_pred, average='weighted')
        precision_weighted = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall_weighted = recall_score(y_true, y_pred, average='weighted')

        # Metriche per classe
        report = pd.DataFrame({
            "precision": precision_score(y_true, y_pred, average=None, zero_division=0),
            "recall": recall_score(y_true, y_pred, average=None)
        }, index=np.unique(y_true))

        roc_auc = roc_auc_score(y_true, y_pred)

        # Assegno le metriche per classe
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

        # Stampa a video
        print(f"\n🔹 Baseline dottore '{col}' vs '{final_col}':")
        log(f"\n🔹 Baseline dottore '{col}' vs '{final_col}':")
        for metric_name, value in metrics.items():
            print(f"  {metric_name}: {value:.4f}" if value is not None else f"  {metric_name}: None")

        results[col] = metrics

        # 🔹 Salvataggio matrice di confusione
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
            print(f"💾 Matrice di confusione salvata in {cm_path}")
            log(f"Matrice di confusione salvata in {cm_path}")

    # Salvataggio CSV
    if output_folder:
        baseline_df = pd.DataFrame([
            {"visit": col, **metrics} for col, metrics in results.items()
        ])
        os.makedirs(output_folder, exist_ok=True)
        baseline_path = os.path.join(output_folder, "baseline_vs_final.csv")
        baseline_df.to_csv(baseline_path, index=False)
        print(f"\n💾 Baseline salvata in {baseline_path}")
        log(f"\n💾 Baseline salvata in {baseline_path}")

    return results
