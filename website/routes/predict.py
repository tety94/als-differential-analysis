from flask import Blueprint, request, jsonify
from sqlalchemy.orm import sessionmaker
from website.models import PatientPrediction
from website.db_connection import engine
import pandas as pd
import joblib
import os
import json
from datetime import datetime
import numpy as np
from website.utilities import clean_for_model, clean_for_db
from website.models import Model
from website.db_connection import engine
from sqlalchemy.orm import sessionmaker
from utilities.models import get_models
from config import features, rename_map

predict_bp = Blueprint('predict', __name__)

Session = sessionmaker(bind=engine)
ML_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), '../models')
pipelines = {}



@predict_bp.route('/predict', methods=['POST'])
def predict():
    form_data = request.get_json()
    print("📦 Dati ricevuti:", form_data)
    Session = sessionmaker(bind=engine)
    session = Session()

    numeric_fields = features['third_level']['numerical_cols']
    categorical_fields = features['third_level']['categorical_columns']

    if 'weight_diagnosis (kg)' in numeric_fields:
        numeric_fields.remove('weight_diagnosis (kg)')

    # crea due dizionari: uno per DB, uno per modello
    db_data = {}
    model_data = {}

    for a in categorical_fields + numeric_fields:
        field = [key for key, value in rename_map.items() if value == a]
        if not field:
            continue
        field = field[0]
        db_data[field] = clean_for_db(form_data.get(field))
        v = form_data.get(field)
        if a in numeric_fields:
            v = clean_for_model(v, field, categorical_fields)
        model_data[field] = v

    # dataframe input per modello
    model_data_renamed = {rename_map.get(k, k): v for k, v in model_data.items()}
    input_df = pd.DataFrame([model_data_renamed])

    for col in categorical_fields + numeric_fields:
        if col not in input_df.columns:
            input_df[col] = pd.NA

    # salva paziente nel DB
    patient = PatientPrediction(**db_data)

    df = input_df.copy()
    for col in df.select_dtypes(bool).columns:
        df[col] = df[col].astype(int)
    record_dict = clean(df.to_dict(orient='records')[0])
    patient.model_input = record_dict

    session.add(patient)
    session.commit()

    # predizione
    results = {}
    model_outputs = {}
    models_to_use = list(get_models().keys())
    model_type = form_data.get('model_type')

    for model_name in models_to_use:
        print('#########################')
        print(model_name)
        print('#########################')
        version = (
            session.query(Model)
            .filter(Model.name == model_name)
            .order_by(Model.id.desc())
            .first()
        ).version
        model_filename = f"{model_type}/{model_name}_{version}.joblib"
        model_path = os.path.join(ML_DIR, model_filename)

        if model_name not in pipelines and os.path.exists(model_path):
            pipelines[model_name] = joblib.load(model_path)
        if model_name not in pipelines:
            continue

        pipe = pipelines[model_name]["pipeline"]
        input_df_model = input_df.copy()

        # riempi le colonne mancanti e riordina
        feature_order = pipelines[model_name]["feature_columns"]
        missing_cols = [c for c in feature_order if c not in input_df_model.columns]
        for c in missing_cols:
            if c in numeric_fields:
                input_df_model[c] = np.nan
            else:
                if model_name == "CatBoost":
                    input_df_model[c] = "missing"
                else:
                    input_df_model[c] = -1
        input_df_model = input_df_model[feature_order]

        # gestione specifica modello
        if model_name == "CatBoost":
            for col in categorical_fields:
                input_df_model[col] = input_df_model[col].fillna("missing").astype(str)
                input_df_model[col] = input_df_model[col].apply(lambda x: str(x) if x not in [None, ""] else "missing")
            input_df_model[numeric_fields] = input_df_model[numeric_fields].astype(float)
        elif model_name == "LightGBM":
            input_df_model[numeric_fields] = input_df_model[numeric_fields].astype(float)
            for col in categorical_fields:
                # sostituisci NaN o stringhe vuote con -1, poi cast a int
                input_df_model[col] = input_df_model[col].replace([None, "", "missing"], -1).astype(int)
        else:
            input_df_model[numeric_fields] = input_df_model[numeric_fields].astype(float)
            for col in categorical_fields:
                input_df_model[col] = input_df_model[col].replace([None, "", "missing"], -1).astype(int)

        # predizione
        pred_class = int(pipe.predict(input_df_model)[0])
        pred_proba = float(pipe.predict_proba(input_df_model)[0][pred_class])  # cast a float standard
        results[model_name] = {"class": pred_class, "probability": pred_proba}

    patient.results = results
    model_outputs["input_summary"] = record_dict
    patient.model_log = model_outputs
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