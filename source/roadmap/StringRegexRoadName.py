import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator


class StringRegexRoadName(BaseEstimator, TransformerMixin):
    def __init__(self, feature_name):
        self.feature_name = feature_name
        self.search_pattern = '(국도|지방도|로|교)'

    def fit(self, x_data, y_data=None):
        return self

    def transform(self, x_data):
        output = x_data[self.feature_name].str.extract(
            self.search_pattern
        )

        output = output.apply(
            lambda x: x if len(x) > 0 else '기타'
        )
        return output