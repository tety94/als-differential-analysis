import sys
import pandas as pd
import joblib
import json
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from python_api.classes.patients_input import PatientInput

# -------------------------
# 1️⃣ Leggi ID passato da PHP
# -------------------------
input_id = int(sys.argv[1])

# -------------------------
# 2️⃣ Connessione DB
# -------------------------
engine = create_engine("mysql+pymysql://user:pass@localhost/dbname")
Session = sessionmaker(bind=engine)
session = Session()

# -------------------------
# 3️⃣ Leggi la riga
# -------------------------
patient_row = session.query(PatientInput).filter_by(id=input_id).first()
if patient_row is None:
    print(json.dumps({"error": "ID non trovato"}))
    sys.exit(1)

# converti in DataFrame
input_df = pd.DataFrame([{
    "age": patient_row.age,
    "sex": patient_row.sex,
    "score1": patient_row.score1,
    "score2": patient_row.score2
}])

# -------------------------
# 4️⃣ Carica pipeline
# -------------------------
pipelines = {
    "RandomForest": joblib.load("ml/models/RandomForest_best.joblib"),
    "LightGBM": joblib.load("ml/models/LightGBM_best.joblib"),
}

# -------------------------
# 5️⃣ Predizione
# -------------------------
results = {}
for name, pipe in pipelines.items():
    pred_class = pipe.predict(input_df)[0]
    pred_proba = pipe.predict_proba(input_df)[0,1]
    results[name] = {"class": int(pred_class), "probability": float(pred_proba)}

# -------------------------
# 6️⃣ Output JSON
# -------------------------
print(json.dumps(results))
