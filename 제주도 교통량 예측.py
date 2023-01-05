import warnings
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
import catboost as cat
import optuna
warnings.filterwarnings('ignore')

from source.base import *
from source.roadmap import *


def pipeline(x_data, y_data):
    all_features = x_data.columns

    _nummeries = make_column_transformer(
        (CycleSineHour('base_hour'), all_features),
        (CycleCosineHour('base_hour'), all_features),
        (RoadLength(['start_latitude', 'start_longitude', 'end_latitude', 'end_longitude']), all_features),
        (DistanceFromLandmark(['start_latitude', 'start_longitude']), all_features),
        # (GroupMeanStatistics('base_hour'), all_features),
    )
    nummeries = make_pipeline(
        _nummeries,
        SimpleImputer(strategy='most_frequent'),
        StandardScaler()
    )

    _categories = make_column_transformer(
        (WeatherSeasonMonth('base_date'), all_features),
        (HasLeaveMonth('base_date'), all_features),
        (HasWeekend('base_date'), all_features),
        (CategoryWeightWeekday('base_date'), all_features),
        (CategoryByTime('base_hour'), all_features),
        (HasOneNodeName(['start_node_name', 'end_node_name']), all_features),
        (HasTurnRestricted(['start_turn_restricted', 'end_turn_restricted']), all_features),
        (StringRegexRoadName('road_name'), all_features),
        (CategoryAreaCluster(['start_latitude', 'start_longitude']), all_features)
    )
    categories = make_pipeline(
        _categories,
        SimpleImputer(strategy='most_frequent'),
        OneHotEncoder(categories='auto', sparse=False, handle_unknown='ignore')
    )

    features = make_column_transformer(
        (nummeries, all_features),
        (categories, all_features)
    )

    _pipeline = make_pipeline(
        features,
        cat.CatBoostRegressor(random_state=27, verbose=False)
    )

    model = 'catboostregressor'

    params = {
        f'{model}__learning_rate': optuna.distributions.LogUniformDistribution(0.01, 0.3),
    }
    search = optuna.integration.OptunaSearchCV(
        _pipeline,
        params,
        cv=3,
        n_trials=5,
        timeout=600,
        scoring='neg_root_mean_squared_error',
    )
    search.fit(x_data, y_data)

    score = search.best_score_
    print(f'best score = {-score: .5f}')

    return search


if __name__ == '__main__':
    data = read_parquet('data/road.parquet')

    categories = sorted(
        data.maximum_speed_limit.unique()
    )
    for category in categories:
        x_train = data[
            data.maximum_speed_limit == category
        ].copy()

        y_train = x_train.pop('target')

        predictor = pipeline(x_train, y_train)

        # predictor.predict(x_train)
