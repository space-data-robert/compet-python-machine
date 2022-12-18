import numpy as np
from sklearn.base import TransformerMixin, BaseEstimator


class CycleCosineHour(BaseEstimator, TransformerMixin):
    def __init__(self, feature_name):
        self.feature_name = feature_name

    def fit(self, x_data, y_data=None):
        return self

    def transform(self, x_data):
        output = x_data[self.feature_name].apply(
            lambda num_value: np.cos(2 * np.pi * (num_value / 24))
        )
        return output[:, None]

