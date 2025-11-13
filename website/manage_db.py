import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models import Base
from db_connection import engine

def migrate_fresh():
    # elimina tutte le tabelle
    Base.metadata.drop_all(engine)
    print("Tabelle cancellate.")

    # ricrea tutte le tabelle
    Base.metadata.create_all(engine)
    print("Tabelle ricreate.")

if __name__ == "__main__":
    migrate_fresh()
