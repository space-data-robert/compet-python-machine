from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from source.base import *
from source.roadmap import *


def pipeline(data):
    all_features = data.columns

    num_feature = make_column_transformer(
        (SelectFeature('maximum_speed_limit'), ['maximum_speed_limit']),
        (CycleSineHour('base_hour'), ['base_hour']),
        (CycleCosineHour('base_hour'), ['base_hour']),
        (GroupMeanStatistics('base_hour', 'target'), ['base_hour', 'target']),
        (RoadLength(['start_latitude', 'start_longitude', 'end_latitude', 'end_longitude']),
         ['start_latitude', 'start_longitude', 'end_latitude', 'end_longitude']),
        (DistanceFromLandmark(['start_latitude', 'start_longitude']),
         ['start_latitude', 'start_longitude'])
    )
    num_pipeline = make_pipeline(
        num_feature,
        StandardScaler()
    )

    cate_feature = make_column_transformer(
        (WeatherSeasonMonth('base_date'), ['base_date']),
        (HasLeaveMonth('base_date'), ['base_date']),
        (HasWeekend('base_date'), ['base_date']),
        (CategoryWeightWeekday('base_date'), ['base_date']),
        (CategoryByTime('base_hour'), ['base_hour']),
        (HasOneNodeName(['start_node_name', 'end_node_name']),
         ['start_node_name', 'end_node_name']),
        (HasTurnRestricted(['start_turn_restricted', 'end_turn_restricted']),
         ['start_turn_restricted', 'end_turn_restricted']),
        (StringRegexRoadName('road_name'), ['road_name']),
        (CategoryAreaCluster(['start_latitude', 'start_longitude']),
         ['start_latitude', 'start_longitude'])
    )
    cate_pipeline = make_pipeline(
        cate_feature,
        OneHotEncoder(categories='auto', sparse=False)
    )

    feat_pipeline = make_column_transformer(
        (num_pipeline, all_features),
        (cate_pipeline, all_features)
    )
    return feat_pipeline

if __name__ == '__main__':
    data = read_parquet('data/road.parquet')
    pipeline = pipeline(data)