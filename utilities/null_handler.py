import pandas as pd
import numpy as np
import logging


def report_nulls(df):
    """Returns a DataFrame with the count and percentage of nulls per column."""
    null_counts = df.isna().sum()
    null_percent = df.isna().mean() * 100
    report = pd.DataFrame({
        'null_count': null_counts,
        'null_percent': null_percent
    }).sort_values('null_percent', ascending=False)
    return report



def impute_nulls(log, df, categorical_cols, threshold_mode=0.5, use_missing_for_cat=True):
    """
    Imputes null values in categorical columns and ensures they are all
    strings. Numeric columns are left untouched (CatBoost handles NaNs
    internally).

    Parameters
    ----------
    log : callable
        Logging function
    df : pd.DataFrame
        DataFrame to impute
    categorical_cols : list
        Categorical columns
    threshold_mode : float
        Minimum share required to use the mode on categorical columns
        (if use_missing_for_cat=False)
    use_missing_for_cat : bool
        If True, nulls in categorical columns become 'missing'
    """
    df = df.copy()
    df.replace(['nan', 'NaN', 'None', ''], np.nan, inplace=True)

    for col in categorical_cols:
        if col not in df.columns:
            continue

        # fill missing values
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

        # convert everything to string, so CatBoost only sees strings
        df[col] = df[col].apply(lambda x: str(int(x)) if isinstance(x, float) and x.is_integer() else str(x))

    return df
