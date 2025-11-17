from flask import Flask, render_template
from website.routes.predict import predict_bp

app = Flask(
    __name__,
    static_url_path='/als-differential-analysis/static',  # URL pubblico
    static_folder='static'                                # cartella fisica relativa a app.py
)

# registra blueprint
app.register_blueprint(predict_bp, url_prefix='/als-differential-analysis')

# route principale
@app.route('/als-differential-analysis/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
