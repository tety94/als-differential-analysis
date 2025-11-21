from sklearn.ensemble import HistGradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from config import random_state


def get_models():
    models = {
        'CatBoost': CatBoostClassifier(
            verbose=0,
            random_state=random_state,
            class_weights='balanced'
        ),

        # 'HistGradientBoosting': HistGradientBoostingClassifier(
        #     max_iter=200,
        #     random_state=random_state
        # ),
        #
        # 'XGBoost': XGBClassifier(
        #     eval_metric="logloss",
        #     random_state=random_state,
        #     n_jobs=-1,
        #     tree_method="hist"     # più veloce, gestisce bene i missing
        # ),
        #
        # 'LightGBM': LGBMClassifier(
        #     class_weight="balanced",
        #     random_state=random_state,
        #     n_jobs=-1,
        #     verbose=-1
        # ),
    }

    return models
