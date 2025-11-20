from sklearn.base import BaseEstimator, ClassifierMixin
from catboost import CatBoostClassifier

class CatBoostWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, iterations=500, learning_rate=0.1, depth=6, cat_features=None):
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.depth = depth
        self.cat_features = cat_features
        self.model = None

    def fit(self, X, y):
        self.model = CatBoostClassifier(
            iterations=self.iterations,
            learning_rate=self.learning_rate,
            depth=self.depth,
            verbose=0
        )
        self.model.fit(X, y, cat_features=self.cat_features)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)
