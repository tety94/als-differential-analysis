from flask import Flask, render_template
from website.routes.predict import predict_bp

app = Flask(
    __name__,
    static_url_path='/als-differential-analysis/static',
    static_folder='website/static'
)

# registra blueprint
app.register_blueprint(predict_bp, url_prefix='/als-differential-analysis')

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
