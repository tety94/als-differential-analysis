import pandas as pd
import numpy as np
from config import csv_path, target_col, t_1_visit
import logging

def load_data(categorical_columns,numerical_cols):
    logging.info(f"Loading data from {csv_path}")
    df = pd.read_csv(csv_path, sep=',', engine='python')
    logging.info(f"Original shape: {df.shape}")

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in CSV")

    # Target binarization
    df = df[~df[target_col].isna()].copy()
    df[target_col] = pd.to_numeric(df[target_col], errors='coerce')
    df = df.dropna(subset=[target_col])
    df[target_col] = (df[target_col] != 0).astype(int)
    df[target_col] = df[target_col].apply(lambda x: 0 if int(x) == 0 else 1)

    # target_medici_1_visit binarization
    df = df[~df[t_1_visit].isna()].copy()
    df[t_1_visit] = pd.to_numeric(df[t_1_visit], errors='coerce')
    df = df.dropna(subset=[t_1_visit])
    df[t_1_visit] = (df[t_1_visit] != 0).astype(int)
    df[t_1_visit] = df[t_1_visit].apply(lambda x: 0 if int(x) == 0 else 1)

    df = df[categorical_columns + numerical_cols + [target_col, t_1_visit]]

    df = df[~df['site_of_onset'].isin([6,7])]

    # in our dataset ALS = 0, other diseases > 0
    df[target_col] = np.where(df[target_col] == 0, 1, 0)
    df[t_1_visit] = np.where(df[t_1_visit] == 0, 1, 0)

    return df
