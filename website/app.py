from flask import Flask, render_template
from website.routes.predict import predict_bp

app = Flask(__name__)

# registra blueprint
app.register_blueprint(predict_bp)

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
