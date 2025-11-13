# website/db_connection.py
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, scoped_session
from config import DB_USER, DB_PASS, DB_HOST, DB_PORT, DB_NAME
import os

# =============================
# CONFIGURAZIONE DATABASE
# =============================

DATABASE_URL = f"mysql+pymysql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# =============================
# CREAZIONE ENGINE E SESSIONE
# =============================

# echo=True → stampa le query SQL nel terminale, utile per debug
engine = create_engine(DATABASE_URL, echo=False)

# Session factory thread-safe
SessionLocal = scoped_session(sessionmaker(bind=engine, autoflush=False, autocommit=False))