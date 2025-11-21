import pandas as pd
import numpy as np
import logging


def report_nulls(df):
    """Restituisce un DataFrame con numero e percentuale di null per colonna."""
    null_counts = df.isna().sum()
    null_percent = df.isna().mean() * 100
    report = pd.DataFrame({
        'null_count': null_counts,
        'null_percent': null_percent
    }).sort_values('null_percent', ascending=False)
    return report



def impute_nulls(log, df, categorical_cols, threshold_mode=0.5, use_missing_for_cat=True):
    """
    Imputa i valori null nelle colonne categoriali e garantisce che tutte siano stringhe.
    Le numeriche non vengono toccate (CatBoost gestisce i NaN internamente).

    Parametri
    ----------
    log : funzione
        Funzione di logging
    df : pd.DataFrame
        DataFrame da imputare
    categorical_cols : list
        Colonne categoriali
    threshold_mode : float
        Percentuale minima per usare la moda su categoriali (se use_missing_for_cat=False)
    use_missing_for_cat : bool
        Se True, i null nelle categoriali diventano 'missing'
    """
    df = df.copy()
    df.replace(['nan', 'NaN', 'None', ''], np.nan, inplace=True)

    for col in categorical_cols:
        if col not in df.columns:
            continue

        # riempi i valori mancanti
        if use_missing_for_cat:
            log(f'{col} Categorical: filling missing with "missing"')
            df[col] = df[col].fillna("missing")
        else:
            mode = df[col].mode()
            if not mode.empty and (df[col].value_counts(normalize=True).iloc[0] >= threshold_mode):
                log(f'{col} Categorical: filling missing with mode')
                df[col] = df[col].fillna(mode[0])
            else:
                log(f'{col} Categorical: filling missing randomly')
                df[col] = df[col].fillna(np.random.choice(df[col].dropna()))

        # converti tutto in stringa, così CatBoost vede solo stringhe
        df[col] = df[col].apply(lambda x: str(int(x)) if isinstance(x, float) and x.is_integer() else str(x))

    return df
