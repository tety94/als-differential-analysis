from flask import Flask, render_template, Blueprint
from website.routes.predict import predict_bp

# --- Blueprint principale per il sito Flask ---
main_bp = Blueprint(
    'main',
    __name__,
    url_prefix='/als-differential-analysis',           # prefisso URL
    static_folder='website/static',                   # percorso fisico statici
    static_url_path='/als-differential-analysis/static'  # URL statici
)

@main_bp.route('/')
def index():
    return render_template('index.html')

# --- Crea l'app Flask ---
app = Flask(
    __name__,
    static_url_path='/als-differential-analysis/static',
    static_folder='website/static'
)

# registra blueprint principale e predict_bp
app.register_blueprint(main_bp)
app.register_blueprint(predict_bp, url_prefix='/als-differential-analysis')

# --- Esegui l'app in locale per testing ---
if __name__ == '__main__':
    app.run(debug=True)
