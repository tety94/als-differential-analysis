# -*- coding: utf-8 -*-
"""
Module: correlation_analysis
Correlation analysis between numeric variables and against the target.
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
    Runs the correlation analysis between numeric variables and against
    the target.

    Parameters
    ----------
    X : pd.DataFrame
        Feature dataset (only numeric columns are considered)
    y : pd.Series or np.array
        Target variable (numeric)
    output_folder : str
        Folder where results are saved
    threshold : float
        Threshold above which two variables are considered strongly
        correlated

    Output
    ------
    - CSV with the correlation matrix
    - CSV with strongly correlated feature pairs
    - CSV with each feature's correlation with the target
    - Correlation heatmap saved as PNG
    """

    log = logging.info
    log("🔍 Starting numeric correlation analysis")

    # Select only numeric columns
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    if not num_cols:
        log("⚠️ No numeric column found to analyze correlations.")
        return

    X_num = X[num_cols].copy()

    # Correlation matrix
    corr_matrix = X_num.corr(method='pearson')
    corr_path = os.path.join(output_folder, "correlation_matrix.csv")
    corr_matrix.to_csv(corr_path)
    log(f"Correlation matrix saved to {corr_path}")

    # Strongly correlated pairs
    strong_corrs = (
        corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        .stack()
        .reset_index()
        .rename(columns={'level_0': 'Feature_1', 'level_1': 'Feature_2', 0: 'Correlation'})
    )
    strong_corrs = strong_corrs[strong_corrs['Correlation'].abs() >= threshold]
    strong_corrs_path = os.path.join(output_folder, "strong_correlations.csv")
    strong_corrs.to_csv(strong_corrs_path, index=False)
    log(f"Strongly correlated pairs saved to {strong_corrs_path}")

    # Correlation with target
    y_series = pd.Series(y).astype(float)
    target_corr = X_num.apply(lambda col: col.corr(y_series))
    target_corr = target_corr.sort_values(ascending=False).rename("Correlation_with_target")
    target_corr_path = os.path.join(output_folder, "feature_target_correlation.csv")
    target_corr.to_csv(target_corr_path)
    log(f"Correlation with target saved to {target_corr_path}")

    # Heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, cmap='coolwarm', center=0)
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    heatmap_path = os.path.join(output_folder, "correlation_heatmap.png")
    plt.savefig(heatmap_path, bbox_inches='tight')
    plt.close()
    log(f"Correlation heatmap saved to {heatmap_path}")

    # Return main results
    return {
        "corr_matrix": corr_matrix,
        "strong_corrs": strong_corrs,
        "target_corr": target_corr
    }


# Drops strongly correlated columns
def drop_strongly_correlated(X, strong_corrs, categorical_cols, numeric_cols):
    """
    Drops one column from each pair of strongly correlated features.
    Chooses to remove the second column of each pair.

    Parameters
    ----------
    X : pd.DataFrame
        Feature dataset
    strong_corrs : pd.DataFrame
        DataFrame with columns ['Feature_1', 'Feature_2', 'Correlation']
    categorical_cols : list
        List of categorical columns
    numeric_cols : list
        List of numeric columns

    Returns
    -------
    X_clean : pd.DataFrame
        Dataset without the strongly correlated columns
    removed_cols : list
        List of removed columns
    numeric_cols_new : list
        Updated numeric column list
    categorical_cols_new : list
        Updated categorical column list
    """
    removed_cols = []

    # Iterate over all strongly correlated pairs
    for _, row in strong_corrs.iterrows():
        col_to_remove = row['Feature_2']
        if col_to_remove in X.columns and col_to_remove not in removed_cols:
            removed_cols.append(col_to_remove)

    # Drop columns
    X_clean = X.drop(columns=removed_cols, errors='ignore')

    # Update numeric and categorical column lists
    numeric_cols_new = [c for c in numeric_cols if c not in removed_cols]
    categorical_cols_new = [c for c in categorical_cols if c not in removed_cols]

    return X_clean, removed_cols, numeric_cols_new, categorical_cols_new
