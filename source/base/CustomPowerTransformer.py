import pandas as pd
from sklearn.preprocessing import PowerTransformer
from sklearn.base import TransformerMixin, BaseEstimator


class CustomPowerTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, inversed=False):
        self.transformer = PowerTransformer()

    def fit(self, X, y=None):
        self.transformer.fit(
            pd.DataFrame(X)
        )
        return self

    def transform(self, X, y=None):
        output = self.transformer.transform(
            pd.DataFrame(X)
        )
        return output

    def inverse_transform(self, X, y=None):
        output = self.transformer.inverse_transform(X)
        return output
    