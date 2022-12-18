import pandas as pd
from haversine import haversine
from sklearn.base import TransformerMixin, BaseEstimator


class DistanceFromLandmark(BaseEstimator, TransformerMixin):
    def __init__(self, feature_names):
        self.feature_names = feature_names
        self.landmark_points = [
            [33.49962, 126.53118],      # 제주시
            [33.25412, 126.56007],      # 서귀포시
            [33.36141, 126.52941],      # 한라산
            [33.45852, 126.94225],      # 성산
            [33.24634, 126.41973]       # 중문
        ]

    def fit(self, x_data, y_data=None):
        return self

    def transform(self, x_data, y_data=None):
        [x_start, y_start] = self.feature_names

        arr_output = pd.DataFrame()
        for enum, landmark_point in enumerate(self.landmark_points):
            output = x_data.apply(
                lambda data: haversine(
                    [data[x_start], data[y_start]],
                    landmark_point), axis=1
            )

            arr_output = pd.concat(
                [arr_output, output], axis=1
            )
        return arr_output