from sklearn.base import TransformerMixin, BaseEstimator


class HasOneNodeName(BaseEstimator, TransformerMixin):
    def __init__(self, feature_names):
        self.feature_names = feature_names

    def fit(self, x_data, y_data=None):
        return self

    def transform(self, x_data):
        [start_name, end_name]= self.feature_names

        output = x_data.apply(
            lambda data: (1 if data[start_name] == data[end_name] else 0), axis=1
        )
        return output[:, None]