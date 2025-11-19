from flask import Flask, render_template, request, session, redirect, url_for
from flask_babel import Babel, _
from website.routes.predict import predict_bp

# --- Crea l'app Flask ---
app = Flask(
    import_name=__name__,
    static_url_path='/als-differential-analysis/static',
    static_folder='static',
    template_folder='templates'
)

app.config['BABEL_DEFAULT_LOCALE'] = 'it'
app.config['BABEL_SUPPORTED_LOCALES'] = ['it', 'en']
app.config['SECRET_KEY'] = 'your-secret-key-here-change-in-production'  # necessario per session


# --- Inizializza Babel ---
def get_locale():
    # 1. URL parameter
    lang = request.args.get('lang')
    if lang in app.config['BABEL_SUPPORTED_LOCALES']:
        session['language'] = lang
        return lang

    # 2. Session
    if 'language' in session:
        return session['language']

    # 3. Accept-Language header
    return request.accept_languages.best_match(app.config['BABEL_SUPPORTED_LOCALES']) or 'it'


babel = Babel(app, locale_selector=get_locale)

# --- Registra blueprint ---
app.register_blueprint(predict_bp, url_prefix='/als-differential-analysis')

# --- Route principale ---
@app.route('/als-differential-analysis/')
def index():
    return render_template('index.html')

@app.context_processor
def inject_language():
    return {"current_lang": session.get("language", "it")}


# --- Cambia lingua ---
@app.route('/als-differential-analysis/set-lang/<lang>')
def set_language(lang):
    if lang in app.config['BABEL_SUPPORTED_LOCALES']:
        session['language'] = lang
    return redirect(request.referrer or url_for('index'))

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8000)
