import numpy as np

def clean_for_model(value, cast_type=float):
    """Valori mancanti diventano np.nan per il modello"""
    try:
        if value is None or value == '':
            return np.nan
        return cast_type(value)
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

