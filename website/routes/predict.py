from flask import Blueprint, request, jsonify
from sqlalchemy.orm import sessionmaker
from website.models import PatientPrediction, Model
from website.db_connection import engine
import pandas as pd
import joblib
import os
from datetime import datetime
import numpy as np
from website.utilities import clean_for_model, clean_for_db
from config import features, rename_map

predict_bp = Blueprint('predict', __name__)

ML_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), '../models')

Session = sessionmaker(bind=engine)


@predict_bp.route('/predict', methods=['POST'])
def predict():
    form_data = request.get_json()
    print("📦 Dati ricevuti:", form_data)

    session = Session()

    numeric_fields = features['third_level']['numerical_cols'].copy()
    categorical_fields = features['third_level']['categorical_columns'].copy()

    # rimuovi peso se presente
    if 'weight_diagnosis (kg)' in numeric_fields:
        numeric_fields.remove('weight_diagnosis (kg)')

    # ─── Preparazione input per modello ───
    db_data = {}
    model_data = {}

    # Cicla su tutte le feature
    for a in categorical_fields + numeric_fields:
        # Recupera il nome originale della colonna nel DB
        field = [key for key, value in rename_map.items() if value == a]
        if not field:
            continue
        field = field[0]

        # Salva valore "pulito" per il DB
        db_data[field] = clean_for_db(form_data.get(field))

        # Prepara valore per il modello
        v = form_data.get(field)

        if a in numeric_fields:
            # Conversione numerica
            v = clean_for_model(v, field, categorical_fields)
        elif a in categorical_fields:
            # Conversione categorica coerente con il training
            v = str(v) if v not in [None, ""] else "missing"

        model_data[field] = v

    # Crea DataFrame input per il modello
    input_df = pd.DataFrame([{rename_map.get(k, k): v for k, v in model_data.items()}])

    # Aggiungi eventuali colonne mancanti
    for col in categorical_fields:
        if col not in input_df.columns:
            input_df[col] = "missing"
    for col in numeric_fields:
        if col not in input_df.columns:
            input_df[col] = np.nan

    # Conversione tipi
    for col in categorical_fields:
        input_df[col] = input_df[col].astype(str)
    input_df[numeric_fields] = input_df[numeric_fields].astype(float)


    # salva paziente nel DB
    patient = PatientPrediction(**db_data)
    df_clean = input_df.copy()
    for col in df_clean.select_dtypes(bool).columns:
        df_clean[col] = df_clean[col].astype(int)
    record_dict = clean(df_clean.to_dict(orient='records')[0])
    patient.model_input = record_dict

    session.add(patient)
    session.commit()

    # ---- Solo CatBoost ----
    model_name = "CatBoost"
    model_type = form_data.get('model_type')

    # carica versione più recente del modello dal DB
    model_record = (
        session.query(Model)
        .filter(Model.name == model_name)
        .order_by(Model.id.desc())
        .first()
    )

    if not model_record:
        session.close()
        return jsonify({"success": False, "error": "Modello CatBoost non trovato."})

    version = model_record.version
    model_path = os.path.join(ML_DIR, f"{model_type}/{model_name}_{version}.joblib")

    if not os.path.exists(model_path):
        session.close()
        return jsonify({"success": False, "error": "File modello non trovato."})

    # carica oggetto CatBoostWrapper salvato come dict con feature_columns
    saved_model_dict = joblib.load(model_path)
    model_obj = saved_model_dict["model"]
    feature_order = saved_model_dict["feature_columns"]


    input_df_model = input_df.copy()

    # riempi le colonne mancanti
    for col in feature_order:
        if col not in input_df_model.columns:
            if col in numeric_fields:
                input_df_model[col] = np.nan
            else:
                input_df_model[col] = "missing"

    # 2. Rimuovi le colonne extra non presenti in feature_order
    input_df_model = input_df_model[feature_order]

    # 3. Assicurati tipi corretti
    for col in categorical_fields:
        if col in input_df_model.columns:
            print(col)
            input_df_model[col] = input_df_model[col].astype(str)
    for col in numeric_fields:
        if col in input_df_model.columns:
            input_df_model[col] = input_df_model[col].astype(float)
    print(feature_order)
    print(input_df_model.columns)
    # exit()

    # predizione
    pred_class = int(model_obj.predict(input_df_model)[0])
    pred_proba = float(model_obj.predict_proba(input_df_model)[0][pred_class])

    results = {model_name: {"class": pred_class, "probability": pred_proba}}

    # salva risultati nel paziente
    patient.results = results
    patient.model_log = {"input_summary": record_dict}
    patient.created_at = datetime.utcnow()

    session.commit()
    session.close()

    return jsonify({"success": True, "results": results})


def clean(obj):
    """Converte numpy, NaN e altri tipi non JS-friendly in oggetti Python standard."""
    if isinstance(obj, dict):
        return {k: clean(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean(v) for v in obj]
    elif isinstance(obj, float) and np.isnan(obj):
        return None
    elif isinstance(obj, np.generic):
        return obj.item()
    else:
        return obj
