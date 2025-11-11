from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier, ExtraTreesClassifier
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from utilities.config import random_state

def get_models():
    models = {
        'LogisticRegression': LogisticRegression(max_iter=1000, random_state=random_state, class_weight='balanced'),
        'RandomForest': RandomForestClassifier(n_estimators=200, random_state=random_state, class_weight='balanced', n_jobs=-1),
        'GradientBoosting': GradientBoostingClassifier(n_estimators=200, random_state=random_state),
        'HistGradientBoosting': HistGradientBoostingClassifier(max_iter=200, random_state=random_state),
        'XGBoost': XGBClassifier(eval_metric="logloss", use_label_encoder=False, random_state=random_state, n_jobs=-1),
        'LightGBM': LGBMClassifier(class_weight="balanced", random_state=random_state, n_jobs=-1),
        'CatBoost': CatBoostClassifier(verbose=0, random_state=random_state, class_weights=[1,2]),
        'ExtraTrees': ExtraTreesClassifier(n_estimators=300, random_state=random_state, class_weight="balanced", n_jobs=-1),
        'LinearSVC': LinearSVC(class_weight="balanced", random_state=random_state)
    }
    return models
