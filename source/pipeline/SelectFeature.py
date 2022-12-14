from sklearn.base import TransformerMixin, BaseEstimator


class SelectFeature(BaseEstimator, TransformerMixin):
    def __init__(self, feature_name):
        self.feature_name = feature_name

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X[self.feature_name]