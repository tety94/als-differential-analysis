from catboost import CatBoostClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np

class CatBoostWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, iterations=500, learning_rate=0.1, depth=6,
                 verbose=0, cat_features=None):
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.depth = depth
        self.verbose = verbose
        self.cat_features = cat_features
        self.model_ = None

    def fit(self, X, y, **fit_params):
        # gestione cat_features passati da fit() o dall'__init__
        cat_features = fit_params.get("cat_features", self.cat_features)

        self.model_ = CatBoostClassifier(
            iterations=self.iterations,
            learning_rate=self.learning_rate,
            depth=self.depth,
            verbose=self.verbose
        )

        self.model_.fit(X, y, cat_features=cat_features)

        # 🔥 necessario per sklearn
        self.classes_ = np.unique(y)

        return self

    def predict(self, X):
        return self.model_.predict(X)

    def predict_proba(self, X):
        return self.model_.predict_proba(X)

    def get_feature_importance(self):
        return self.model_.get_feature_importance()
