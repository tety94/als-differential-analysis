# -*- coding: utf-8 -*-
"""
Modulo: correlation_analysis
Analisi delle correlazioni tra le variabili numeriche e rispetto al target.
"""

import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import logging


def correlation_analysis(X, y, output_folder, threshold=0.8):
    """
    Esegue l'analisi delle correlazioni tra le variabili numeriche e col target.

    Parametri
    ----------
    X : pd.DataFrame
        Dataset delle feature (solo colonne numeriche vengono considerate)
    y : pd.Series o np.array
        Variabile target (numerica)
    output_folder : str
        Cartella in cui salvare i risultati
    threshold : float
        Soglia oltre la quale due variabili sono considerate fortemente correlate

    Output
    ------
    - CSV con la matrice di correlazione
    - CSV con le coppie di feature fortemente correlate
    - CSV con la correlazione di ogni feature con l'output
    - Heatmap delle correlazioni salvata come PNG
    """

    log = logging.info
    log("🔍 Avvio analisi delle correlazioni numeriche")

    # Seleziona solo le colonne numeriche
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    if not num_cols:
        log("⚠️ Nessuna colonna numerica trovata per analizzare le correlazioni.")
        return

    X_num = X[num_cols].copy()

    # Matrice di correlazione
    corr_matrix = X_num.corr(method='pearson')
    corr_path = os.path.join(output_folder, "correlation_matrix.csv")
    corr_matrix.to_csv(corr_path)
    log(f"Matrice di correlazione salvata in {corr_path}")

    # Coppie fortemente correlate
    strong_corrs = (
        corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        .stack()
        .reset_index()
        .rename(columns={'level_0': 'Feature_1', 'level_1': 'Feature_2', 0: 'Correlation'})
    )
    strong_corrs = strong_corrs[strong_corrs['Correlation'].abs() >= threshold]
    strong_corrs_path = os.path.join(output_folder, "strong_correlations.csv")
    strong_corrs.to_csv(strong_corrs_path, index=False)
    log(f"Coppie fortemente correlate salvate in {strong_corrs_path}")

    # Correlazione col target
    y_series = pd.Series(y).astype(float)
    target_corr = X_num.apply(lambda col: col.corr(y_series))
    target_corr = target_corr.sort_values(ascending=False).rename("Correlation_with_target")
    target_corr_path = os.path.join(output_folder, "feature_target_correlation.csv")
    target_corr.to_csv(target_corr_path)
    log(f"Correlazione con il target salvata in {target_corr_path}")

    # Heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, cmap='coolwarm', center=0)
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    heatmap_path = os.path.join(output_folder, "correlation_heatmap.png")
    plt.savefig(heatmap_path, bbox_inches='tight')
    plt.close()
    log(f"Heatmap delle correlazioni salvata in {heatmap_path}")

    # Ritorna risultati principali
    return {
        "corr_matrix": corr_matrix,
        "strong_corrs": strong_corrs,
        "target_corr": target_corr
    }


# Rimuove colonne fortemente correlate
def drop_strongly_correlated(X, strong_corrs, categorical_cols, numeric_cols):
    """
    Rimuove una colonna per ogni coppia di feature fortemente correlate.
    Sceglie di rimuovere la seconda colonna di ogni coppia.

    Parametri
    ----------
    X : pd.DataFrame
        Dataset delle feature
    strong_corrs : pd.DataFrame
        DataFrame con colonne ['Feature_1', 'Feature_2', 'Correlation']
    categorical_cols : list
        Lista delle colonne categoriche
    numeric_cols : list
        Lista delle colonne numeriche

    Ritorna
    -------
    X_clean : pd.DataFrame
        Dataset senza le colonne fortemente correlate
    removed_cols : list
        Lista delle colonne rimosse
    numeric_cols_new : list
        Lista colonne numeriche aggiornate
    categorical_cols_new : list
        Lista colonne categoriche aggiornate
    """
    removed_cols = []

    # Itera su tutte le coppie fortemente correlate
    for _, row in strong_corrs.iterrows():
        col_to_remove = row['Feature_2']
        if col_to_remove in X.columns and col_to_remove not in removed_cols:
            removed_cols.append(col_to_remove)

    # Drop colonne
    X_clean = X.drop(columns=removed_cols, errors='ignore')

    # Aggiorna liste di colonne numeriche e categoriche
    numeric_cols_new = [c for c in numeric_cols if c not in removed_cols]
    categorical_cols_new = [c for c in categorical_cols if c not in removed_cols]

    return X_clean, removed_cols, numeric_cols_new, categorical_cols_new

