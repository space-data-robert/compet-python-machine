import numpy as np
from sklearn.cluster import KMeans
from sklearn.base import TransformerMixin, BaseEstimator


class CategoryAreaCluster(BaseEstimator, TransformerMixin):
    def __init__(self, feature_names):
        self.feature_names = feature_names
        self.manual_centres = np.array([
            [33.26345, 126.52038], [33.37082, 126.29767],
            [33.48077, 126.49467], [33.41815, 126.77398]
        ])
        self.cluster = KMeans(
            n_clusters=len(self.manual_centres),
            init=self.manual_centres,
            random_state=27
        )

    def fit(self, x_data, y_data=None):
        self.cluster.fit(
            x_data[self.feature_names]
        )
        return self

    def transform(self, x_data):
        output = self.cluster.predict(
            x_data[self.feature_names]
        ).astype(str)

        return output[:, None]