from sklearn.base import TransformerMixin, BaseEstimator


class HasTurnRestricted(BaseEstimator, TransformerMixin):
    def __init__(self, feature_names):
        self.feature_names = feature_names

    def fit(self, x_data, y=None):
        return self

    def transform(self, x_data):
        [start_restricted, end_restricted]= self.feature_names

        output = x_data.apply(
            lambda data: (1 if data[start_restricted] == data[end_restricted] else 0), axis=1
        )
        return output[:, None]