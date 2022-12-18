import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator


class HasWeekend(BaseEstimator, TransformerMixin):
    def __init__(self, feature_name):
        self.feature_name = feature_name
        self.feature_format = '%Y%m%d'

    def fit(self, x_data, y_data=None):
        return self

    def transform(self, x_data):

        output = pd.to_datetime(
            x_data[self.feature_name],
            format=self.feature_format
        ).dt.weekday

        output = output.apply(
            lambda num_value: 1 if num_value > 4 else 0
        )
        return output[:, None]