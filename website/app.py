from flask import Flask, render_template
from website.routes.predict import predict_bp  # il tuo blueprint predict

# --- Crea l'app Flask ---
app = Flask(
    __name__,
    static_url_path='/als-differential-analysis/static',  # URL pubblico statici
    static_folder='static'                                # cartella fisica relativa a questo file
)

# --- Registra blueprint ---
app.register_blueprint(predict_bp, url_prefix='/als-differential-analysis')

# --- Route principale ---
@app.route('/als-differential-analysis/')
def index():
    return render_template('index.html')


if __name__ == '__main__':
    # solo per test locale
    app.run(debug=True, host='0.0.0.0', port=8000)
