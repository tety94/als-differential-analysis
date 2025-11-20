import numpy as np

def clean_for_model(value, column_name=None, categorical_cols=None):
    """
    Trasforma valori mancanti in np.nan per il modello.
    Se la colonna è categoriale, lascia il valore così com'è.
    """
    try:
        if value is None or value == '':
            return np.nan
        if categorical_cols and column_name in categorical_cols:
            return value  # lascia invariato per categorie
        return float(value)  # cast a float per le altre colonne
    except (ValueError, TypeError):
        return np.nan



def clean_for_db(value, cast_type=float):
    """Valori mancanti diventano None per il DB"""
    try:
        if value is None or value == '':
            return None
        return cast_type(value)
    except (ValueError, TypeError):
        return None

