import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator


class CategoryWeightWeekday(BaseEstimator, TransformerMixin):
    def __init__(self, feature_name):
        self.feature_name = feature_name
        self.feature_format = '%Y%m%d'

    def fit(self, x_data, y_data=None):
        return self

    def transform(self, x_data):
        def weight(num_value):
            if num_value > 4:
                return 3
            if num_value in [0, 4]:
                return 2
            return 1

        output = pd.to_datetime(
            x_data[self.feature_name],
            format=self.feature_format
        ).dt.weekday

        output = output.apply(
            lambda num_value: weight(num_value)
        )
        return output[:, None]