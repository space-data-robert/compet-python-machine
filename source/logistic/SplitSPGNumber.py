from sklearn.base import TransformerMixin, BaseEstimator


class SplitSPGNumber(BaseEstimator, TransformerMixin):
    def __init__(self, feature_name):
        self.feature_name = feature_name
        self.categories = [0, 5, 9, 10, 15]

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        outout = X[[self.feature_name]].copy().astype(str)

        if self.feature_name.startswith('REC'):
            for ind in range(16):
                outout[f'{self.feature_name}{ind}'] = outout[self.feature_name].apply(
                    lambda x: int(x[ind])
                ).astype('category')
            outout.drop(
                self.feature_name, axis=1, inplace=True)

        if self.feature_name.startswith('SEND'):
            for ind in range(len(self.categories) - 1):
                outout[f'{self.feature_name}{ind}'] = outout[self.feature_name].apply(
                    lambda x: sum(
                        list(map(int, x[self.categories[ind]: self.categories[ind + 1]]))
                    )

                ).astype('category')
            outout.drop(
                self.feature_name, axis=1, inplace=True)
        return outout
