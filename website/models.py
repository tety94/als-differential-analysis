from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, Text, JSON
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime

Base = declarative_base()

class PatientPrediction(Base):
    __tablename__ = "patient_predictions"

    id = Column(Integer, primary_key=True, autoincrement=True)

    # input dal form
    site_of_onset = Column(Integer)
    phenotype = Column(Integer)
    type_onset = Column(Integer)
    side_of_onset = Column(Integer)
    elescorial_class = Column(Integer)
    diagn_1_vis = Column(Integer)
    age_onset = Column(Integer)
    delay = Column(Integer)
    hbw = Column(Float)
    weight_diagnosis = Column(Float)
    delta_weight = Column(Float)
    alsfrs_evalutation_1_visit = Column(Float)
    delta_alsfrs = Column(Float)
    alsfrs_bulb = Column(Float)
    alsfrs_aass = Column(Float)
    alsfrs_aaii = Column(Float)
    alsfrs_resp = Column(Float)
    fvc = Column(Float)
    ck = Column(Float)
    turin_tot = Column(Float)
    turin_lower_mn = Column(Float)

    # radio buttons
    n_site = Column(Integer)
    sex = Column(String(10))

    # checkboxes
    familiarity = Column(Boolean)
    second_opinion = Column(Boolean)
    em_lability = Column(Boolean)
    cramps = Column(Boolean)
    fasciculation = Column(Boolean)
    progression = Column(Boolean)
    emg = Column(Boolean)
    eng = Column(Boolean)
    igm_borellina = Column(Boolean)
    igg_borellina = Column(Boolean)
    wb_borellina = Column(Boolean)
    brain_mri_mnd = Column(Boolean)
    brain_mri_other = Column(Boolean)
    spine_mri = Column(Boolean)
    pet = Column(Boolean)
    test_neuro = Column(Boolean)
    genetic_status = Column(Boolean)
    tongue_atrophy = Column(Boolean)

    # risultati dei modelli salvati come JSON (string)
    results = Column(JSON)
    model_input = Column(JSON)
    model_log = Column(JSON)

    created_at = Column(DateTime, default=datetime.utcnow)


class Model(Base):
    __tablename__ = "models"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(255), nullable=False)
    version = Column(String(255), nullable=False)
    params = Column(JSON, nullable=True)
    type = Column(String(255), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
