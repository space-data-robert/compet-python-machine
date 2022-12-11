import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator


class SplitDate(BaseEstimator, TransformerMixin):
    def __init__(self, feature_name):
        self.feature_name = feature_name

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        output = X.copy()

        datetime = pd.to_datetime(
            output[self.feature_name], format='%Y%m%d'
        )
        output['year'] = datetime.dt.month.astype(str)
        output['month'] = datetime.dt.month.astype(str)

        output.drop(self.feature_name, axis=1, inplace=True)

        return output