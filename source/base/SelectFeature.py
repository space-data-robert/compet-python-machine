from sklearn.base import TransformerMixin, BaseEstimator


class SelectFeature(BaseEstimator, TransformerMixin):
    def __init__(self, feature_names):
        self.feature_names = feature_names

    def fit(self, x_data, y_data=None):
        return self

    def transform(self, x_data):
        return x_data[self.feature_names][:, None]