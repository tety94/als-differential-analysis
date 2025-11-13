from flask import Blueprint, request, jsonify
from sqlalchemy.orm import sessionmaker
from website.models import Base, PatientPrediction
from website.db_connection import engine  # connessione centralizzata
import pandas as pd
import joblib
import os
import json
from datetime import datetime
from website.utilities import clean_numeric
from config import version

predict_bp = Blueprint('predict', __name__)

Session = sessionmaker(bind=engine)
ML_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), '../models')
pipelines = {}

@predict_bp.route('/predict', methods=['POST'])
def predict():
    # dati inviati dal form come JSON
    form_data = request.get_json()
    print("📦 Dati ricevuti:", form_data)

    session = Session()

    # Campi numerici
    numeric_fields = [
        'age_onset', 'delay', 'hbw', 'weight_diagnosis', 'delta_weight',
        'alsfrs_evalutation_1_visit', 'delta_alsfrs', 'alsfrs_bulb',
        'alsfrs_aass', 'alsfrs_aaii', 'alsfrs_resp', 'fvc', 'ck',
        'turin_tot', 'turin_lower_mn'
    ]

    for field in numeric_fields:
        form_data[field] = clean_numeric(form_data.get(field), float)

    categorical_fields = [
        'site_of_onset', 'phenotype', 'type_onset', 'side_of_onset',
        'elescorial_class', 'diagn_1_vis', 'n_site', 'sex'
    ]
    for field in categorical_fields:
        form_data[field] = clean_numeric(form_data.get(field), float)

    # Campi booleani (checkbox)
    bool_fields = [
        'familiarity', 'second_opinion', 'em_lability', 'cramps',
        'fasciculation', 'progression', 'emg', 'eng', 'igm_borellina',
        'igg_borellina', 'wb_borellina', 'brain_mri_mnd', 'brain_mri_other',
        'spine_mri', 'pet', 'test_neuro', 'genetic_status', 'tongue_atrophy'
    ]
    for field in bool_fields:
        form_data[field] = bool(form_data.get(field))

    radio_fields = ["n_site", "sex"]

    for rf in radio_fields:
        val = form_data.get(rf)
        print(val)
        exit()
        if val is None or val == "":
            form_data[rf] = None
        else:
            try:
                form_data[rf] = int(val)
            except ValueError:
                form_data[rf] = val

    # crea oggetto paziente e salva DB
    patient = PatientPrediction(**form_data)
    session.add(patient)
    session.commit()

    # prepara dataframe per predizione
    input_df = pd.DataFrame([form_data])

    # lista dei modelli che vuoi usare
    models_to_use = ["LogisticRegression","RandomForest", "GradientBoosting",
                     "HistGradientBoosting", "XGBoost", "LightGBM", "CatBoost", "ExtraTrees"]

    results = {}
    for model_name in models_to_use:
        model_filename = f"{model_name}_{version}.joblib"
        model_path = os.path.join(ML_DIR, model_filename)
        # carica il modello se non è già in memoria
        if model_name not in pipelines and os.path.exists(model_path):
            model_dict = joblib.load(model_path)
            pipelines[model_name] = model_dict

        if model_name not in pipelines:
            continue

        # prendi pipeline e feature columns dal dizionario
        pipe = pipelines[model_name]["pipeline"]
        feature_cols = pipelines[model_name]["feature_columns"]

        # riallinea il dataframe alle colonne usate in training
        input_df_aligned = input_df.reindex(columns=feature_cols, fill_value=0)

        # predizione
        pred_class = int(pipe.predict(input_df_aligned)[0])
        try:
            pred_proba = float(pipe.predict_proba(input_df_aligned)[0, 1])
        except AttributeError:
            # alcuni modelli (es. HistGradientBoosting) usano predict_proba direttamente
            pred_proba = float(pipe.predict_proba(input_df_aligned)[0][1])

        results[model_name] = {"class": pred_class, "probability": pred_proba}

    # salva risultati come JSON
    patient.results = json.dumps(results)
    patient.result_ok = 0 #TODO: mettere media dei valori degli algoritmi
    patient.result_ko = 0 #TODO: mettere media dei valori degli algoritmi
    patient.created_at = datetime.utcnow()
    session.commit()
    session.close()

    return jsonify({"success": True, "results": results})
