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
    familiarity = Column(Boolean, default=False)
    second_opinion = Column(Boolean, default=False)
    em_lability = Column(Boolean, default=False)
    cramps = Column(Boolean, default=False)
    fasciculation = Column(Boolean, default=False)
    progression = Column(Boolean, default=False)
    emg = Column(Boolean, default=False)
    eng = Column(Boolean, default=False)
    igm_borellina = Column(Boolean, default=False)
    igg_borellina = Column(Boolean, default=False)
    wb_borellina = Column(Boolean, default=False)
    brain_mri_mnd = Column(Boolean, default=False)
    brain_mri_other = Column(Boolean, default=False)
    spine_mri = Column(Boolean, default=False)
    pet = Column(Boolean, default=False)
    test_neuro = Column(Boolean, default=False)
    genetic_status = Column(Boolean, default=False)
    tongue_atrophy = Column(Boolean, default=False)

    # risultati dei modelli salvati come JSON (string)
    results = Column(JSON)
    model_input = Column(JSON)
    model_log = Column(JSON)

    created_at = Column(DateTime, default=datetime.utcnow)


class Model(Base):
    __tablename__ = "models"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String, nullable=False)
    version = Column(String, nullable=False)
    params = Column(JSON, nullable=True)
    type = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
