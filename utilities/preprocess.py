import pandas as pd
import numpy as np
import re

def convert_plus_minus(df):
    def conv(x):
        if isinstance(x, str):
            match = re.fullmatch(r"(\d+(?:\.\d+)?)([+-])", x.strip())
            if match:
                base = float(match.group(1))
                return base + 0.25 if match.group(2) == '+' else base - 0.25
        return x
    return df.map(conv)

def separate_columns(df, forced_numerical=[], forced_categorical=[], binary_cols=[]):
    numeric_cols, categorical_cols = [], []
    df = df.replace(',', '.', regex=True)
    for col in df.columns:
        s = df[col].astype(str).str.strip().replace({'': np.nan, 'nan': np.nan})
        if col in forced_categorical:
            df[col] = s.astype(str)
            categorical_cols.append(col)
            continue
        if col in forced_numerical or col in binary_cols:
            df[col] = pd.to_numeric(s, errors='coerce')
            numeric_cols.append(col)
            continue
        # logica automatica
        converted = pd.to_numeric(s, errors='coerce')
        numeric_ratio = converted.notna().mean()
        if numeric_ratio > 0.5 and df[col].nunique() > 5:
            df[col] = converted
            numeric_cols.append(col)
        else:
            df[col] = s.astype(str)
            categorical_cols.append(col)
    return df, numeric_cols, categorical_cols
