from flask import Flask, render_template
from website.routes.predict import predict_bp

# App Flask
app = Flask(
    __name__,
    static_url_path='/als-differential-analysis/static',  # URL pubblico statici
    static_folder='static'                                # cartella fisica statici, relativa a app.py
)

# Registrazione blueprint predict
app.register_blueprint(predict_bp, url_prefix='/als-differential-analysis')

# Route principale
@app.route('/als-differential-analysis/')
def index():
    return render_template('index.html')
