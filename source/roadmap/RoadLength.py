from haversine import haversine
from sklearn.base import TransformerMixin, BaseEstimator


class RoadLength(BaseEstimator, TransformerMixin):
    def __init__(self, feature_names):
        self.feature_names = feature_names

    def fit(self, x_data, y_data=None):
        return self

    def transform(self, x_data):
        [x_start, y_start, x_end, y_end] = self.feature_names

        output = x_data.apply(
            lambda data: haversine(
                [data[x_start], data[y_start]],
                [data[x_end], data[y_end]]), axis=1
        )
        return output[:, None]