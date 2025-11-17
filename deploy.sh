#!/bin/bash
set -e  # esci se qualche comando fallisce

# --- Configurazione ---
APP_DIR="/srv/python-projects/als-differential-analysis"
VENV_DIR="$APP_DIR/venv"
SERVICE_NAME="als-differential-analysis"
GIT_BRANCH="main"   # o il branch che usi

# --- Passa alla cartella dell'app ---
cd "$APP_DIR"

echo "Attivazione virtualenv..."
source "$VENV_DIR/bin/activate"

# --- Aggiorna codice da Git ---
echo "Aggiornamento repository..."
git fetch origin
git reset --hard origin/$GIT_BRANCH

# --- Installa nuove dipendenze se presenti ---
if [ -f requirements.txt ]; then
    echo "Installazione/aggiornamento dipendenze..."
    pip install -r requirements.txt
fi

# --- Riavvia Gunicorn tramite systemd ---
echo "Riavvio servizio Gunicorn..."
sudo systemctl restart "$SERVICE_NAME"

# --- Pulizia e conferma ---
echo "Deploy completato!"
echo "Stato servizio:"
sudo systemctl status "$SERVICE_NAME" --no-pager
