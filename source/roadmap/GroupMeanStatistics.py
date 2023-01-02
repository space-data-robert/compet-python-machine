import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator


class GroupMeanStatistics(BaseEstimator, TransformerMixin):
    def __init__(self, feature_name):
        self.feature_name = feature_name
        self.target_name = 'target'
        self.stats_dictionary = 0

    def fit(self, x_data, y_data=None):
        self.stats_dictionary = x_data.groupby(
            self.feature_name
        )[self.target_name].mean().to_dict()
        return self

    def transform(self, x_data):
        output = x_data[self.feature_name].map(
            self.stats_dictionary
        )
        return output[:, None]