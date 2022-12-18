import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator


class HasLeaveMonth(BaseEstimator, TransformerMixin):
    def __init__(self, feature_name):
        self.feature_name = feature_name
        self.feature_format = '%Y%m%d'
        self.manual_leave = [6, 7, 1, 2]

    def fit(self, x_data, y_data=None):
        return self

    def transform(self, x_data):
        output = pd.to_datetime(
            x_data[self.feature_name],
            format=self.feature_format
        ).dt.month

        output = output.apply(
            lambda num_value: (1 if num_value in self.manual_leave else 0)
        )

        return output[:, None]
