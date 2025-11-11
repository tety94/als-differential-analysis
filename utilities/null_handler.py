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


def impute_nulls(log, df, categorical_cols, numerical_cols, threshold_mode=0.5):
    """
    Imputa i valori null in base al tipo di colonna.
    - categorical_cols: lista colonne categoriali
    - numerical_cols: lista colonne numeriche
    - threshold_mode: percentuale minima per usare la moda su categoriche
    """
    df = df.copy()
    df.replace(['nan', 'NaN', 'None', ''], np.nan, inplace=True)

    # Categorical / binary
    for col in categorical_cols:
        if col in df.columns:
            mode = df[col].mode()
            if not mode.empty and (df[col].value_counts(normalize=True).iloc[0] >= threshold_mode):
                log(f'{col} Categorical: mode')
                df[col] = df[col].fillna(mode[0])
            else:
                # Se nessun valore domina, campiona casualmente dai valori esistenti
                log(f'{col} Categorical: random')
                if(col == 'phenotype (1-8)'):
                    print(col)
                df[col] = df[col].fillna(np.random.choice(df[col].dropna()))

    # Numerical
    for col in numerical_cols:
        if col in df.columns:
            # Se pochi nulli, usa media
            null_ratio = df[col].isna().mean()
            if null_ratio < 0.1:
                log(f'{col} Numerical: mean')
                df[col] = df[col].fillna(df[col].mean())
            else:
                # Altrimenti campiona dai valori esistenti
                log(f'{col} Numerical: random')
                df[col] = df[col].fillna(np.random.choice(df[col].dropna()))

    return df
