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

    numeric_fields = [
        'age_onset', 'delay', 'hbw', 'weight_diagnosis', 'delta_weight',
        'alsfrs_evalutation_1_visit', 'delta_alsfrs', 'alsfrs_bulb',
        'alsfrs_aass', 'alsfrs_aaii', 'alsfrs_resp', 'fvc', 'ck',
        'turin_tot', 'turin_lower_mn'
    ]
    categorical_fields = [
        'site_of_onset', 'phenotype', 'type_onset', 'side_of_onset',
        'elescorial_class', 'diagn_1_vis', 'n_site', 'sex',
        'familiarity', 'second_opinion', 'em_lability', 'cramps',
        'fasciculation', 'progression', 'emg', 'eng', 'igm_borellina',
        'igg_borellina', 'wb_borellina', 'brain_mri_mnd', 'brain_mri_other',
        'spine_mri', 'pet', 'test_neuro', 'genetic_status', 'tongue_atrophy'

    ]
    bool_fields = [
    ]

    # crea due dizionari: uno per DB, uno per modello
    db_data = {}
    model_data = {}

    for field in numeric_fields + categorical_fields:
        db_data[field] = clean_for_db(form_data.get(field))
        model_data[field] = clean_for_model(form_data.get(field))

    for field in bool_fields:
        val = bool(form_data.get(field))
        db_data[field] = val
        model_data[field] = int(val)

    # salva paziente nel DB
    patient = PatientPrediction(**db_data)
    session.add(patient)
    session.commit()

    # predizione
    rename_map = {
        'age_onset': 'age_onset (y)',
        'delay': 'delay (m)',
        'hbw': 'HBW (kg)',
        'weight_diagnosis': 'weight_diagnosis',
        'alsfrs_evalutation_1_visit': 'alsfrs_evalutation_1_visit',
        'delta_alsfrs': 'Δalsfrs',
        'alsfrs_bulb': 'alsfrs_bulb',
        'alsfrs_aass': 'alsfrs_aass',
        'alsfrs_aaii': 'alsfrs_aaii',
        'alsfrs_resp': 'alsfrs_resp.',
        'fvc': 'fvc (%)',
        'ck': 'ck (valore)',
        'turin_tot': 'Turin_tot',
        'turin_lower_mn': 'Turin_lower_MN',
        'site_of_onset': 'site_of_onset',
        'phenotype': 'phenotype (1-8)',
        'type_onset': 'type_onset (0-4)',
        'side_of_onset': 'side_of_onset',
        'elescorial_class': 'elescorial_class (0-3)',
        'diagn_1_vis': 'diagn_1_vis',
        'sex': 'sex (M/F)',
        'familiarity': 'familiarity_MND/demenza/psich./Paget (0/1)',
        'second_opinion': 'second_opinion (0/1)',
        'em_lability': 'em_lability (0/1)',
        'cramps': 'cramps (0/1)',
        'fasciculation': 'fasciculation (0/1)',
        'progression': 'progression (0/1)',
        'emg': 'emg (0/1)',
        'eng': 'eng (0/1)',
        'igm_borellina': 'IgM Borrelia (0/1)',
        'igg_borellina': 'IgG Borrelia (0/1)',
        'wb_borellina': 'WB_Borrelia (0/1)',
        'brain_mri_mnd': 'brain_mri_mnd (0/1)',
        'brain_mri_other': 'brain_mri_other (0/1)',
        'spine_mri': 'spine_mri (0/1)',
        'pet': 'pet (0/1)',
        'test_neuro': 'test_neuro (0/1)',
        'genetic_status': 'genetic_status',
        'tongue_atrophy': 'tongue_atrophy',
        'delta_weight' :'Δweight (kg)',
        'n_site' : 'n_site (0/4)',
    }

    model_data_renamed = {rename_map.get(k, k): v for k, v in model_data.items()}
    input_df = pd.DataFrame([model_data_renamed])

    results = {}
    model_outputs = {}

    models_to_use = ["LogisticRegression", "RandomForest", "GradientBoosting", "HistGradientBoosting",
                     "XGBoost", "LightGBM", "CatBoost", "ExtraTrees"]

    model_type = form_data.get('model_type')
    for model_name in models_to_use:

        version = (session.query(Model)
                   .filter(Model.name == model_name)
                   .order_by(Model.id.desc())
                   .first()).version
        model_filename = f"{model_type}/{model_name}_{version}.joblib"
        model_path = os.path.join(ML_DIR, model_filename)

        # carica modello se non già in memoria
        if model_name not in pipelines and os.path.exists(model_path):
            pipelines[model_name] = joblib.load(model_path)
        if model_name not in pipelines:
            continue

        pipe = pipelines[model_name]["pipeline"]

        # passiamo direttamente i dati grezzi alla pipeline, senza riallineamento manuale
        pred_class = int(pipe.predict(input_df)[0])
        pred_proba = pipe.predict_proba(input_df)[0][pred_class]

        results[model_name] = {"class": pred_class, "probability": pred_proba}

        model_outputs[model_name] = {
            "pred_class": pred_class,
            "pred_proba": pred_proba,
            "timestamp": datetime.utcnow().isoformat(),
            # "user_id": current_user.id,
            "feature_names": list(input_df.columns),
            # "model_version": model.version,
            # "model_params": model.get_params(),
        }

        if hasattr(pipe, "named_steps"):
            transformed = pipe.named_steps["pre"].transform(input_df)
            model_outputs[model_name]["transformed_input"] = transformed.tolist()

        # Pulisce i dati da NaN e tipi numpy
        model_outputs[model_name] = clean(model_outputs[model_name])

    # salva risultati nel DB
    patient.results = results

    df = input_df.copy()
    for col in df.select_dtypes(bool).columns:
        df[col] = df[col].astype(int)
    record_dict = clean(df.to_dict(orient='records')[0])
    patient.model_input = record_dict

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