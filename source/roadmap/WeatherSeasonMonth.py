import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator


class WeatherSeasonMonth(BaseEstimator, TransformerMixin):
    def __init__(self, feature_name):
        self.feature_name = feature_name
        self.feature_format = '%Y%m%d'

    def fit(self, x_data, y_data=None):
        return self

    def transform(self, x_data):
        def season(num_value):
            if num_value // 3 >= 4:
                return str(0)
            return str(num_value // 3)

        output = pd.to_datetime(
            x_data[self.feature_name],
            format=self.feature_format
        ).dt.month

        output = output.apply(
            lambda num_value: season(num_value)
        )
        return output[:, None]