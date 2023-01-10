from sklearn.base import TransformerMixin, BaseEstimator


class HouseholdRate(BaseEstimator, TransformerMixin):
    def __init__(self, total_house_count, empty_house_count):
        self.total_house_count = total_house_count
        self.empty_house_count = empty_house_count

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return (X[self.total_house_count] - X[self.empty_house_count]) / X[
            self.total_house_count
        ]
