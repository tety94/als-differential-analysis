import pandas as pd
from utilities.config import csv_path, target_col, id_cols, t_1_visit
import logging

def load_data():
    logging.info(f"Caricamento dati da {csv_path}")
    df = pd.read_csv(csv_path, sep=',', engine='python')
    logging.info(f"Shape originale: {df.shape}")

    if target_col not in df.columns:
        raise ValueError(f"Colonna target '{target_col}' non trovata nel CSV")

    # Binarizzazione target
    df = df[~df[target_col].isna()].copy()
    df[target_col] = df[target_col].apply(lambda x: 0 if int(x) == 0 else 1)

    # Binarizzazione target_medici_1_visit
    df = df[~df[t_1_visit].isna()].copy()
    df[t_1_visit] = df[t_1_visit].apply(lambda x: 0 if int(x) == 0 else 1)


    return df
