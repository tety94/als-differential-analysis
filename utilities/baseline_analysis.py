import pandas as pd
import logging
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

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

        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "f1": f1_score(y_true, y_pred, average='weighted'),
            "precision": precision_score(y_true, y_pred, average='weighted', zero_division=0),
            "recall": recall_score(y_true, y_pred, average='weighted')
        }

        # Stampa a video
        print(f"\n🔹 Baseline dottore '{col}' vs '{final_col}':")
        log(f"\n🔹 Baseline dottore '{col}' vs '{final_col}':")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")

        results[col] = metrics

    # Salvataggio CSV
    if output_folder:
        baseline_df = pd.DataFrame([
            {"visit": col, **metrics} for col, metrics in results.items()
        ])
        baseline_path = f"{output_folder}/baseline_vs_final.csv"
        baseline_df.to_csv(baseline_path, index=False)
        print(f"\n💾 Baseline salvata in {baseline_path}")
        log(f"\n💾 Baseline salvata in {baseline_path}")

    return results
