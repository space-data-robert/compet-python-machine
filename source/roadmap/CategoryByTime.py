from sklearn.base import TransformerMixin, BaseEstimator


class CategoryByTime(BaseEstimator, TransformerMixin):
    def __init__(self, feature_name):
        self.feature_name = feature_name
        self.manual_bin = [3, 8, 18, 22]

    def fit(self, x_data, y=None):
        return self

    def transform(self, x_data):
        def category(num_value):     
            for enum, bin in enumerate(self.manual_bin):
                if num_value > bin:
                    continue
                return str(enum)
            return str(0)

        output = x_data[self.feature_name].apply(
            lambda num_value: category(num_value)
        )
        return output[:, None]