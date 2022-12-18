from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from source.common import *
from source.roadmap import *


def pipeline():
    feature = make_column_transformer(
        (WeatherSeasonMonth('base_date'), ['base_date']),
        (HasLeaveMonth('base_date'), ['base_date']),
        (HasWeekend('base_date'), ['base_date']),
        (CategoryWeightWeekday('base_date'), ['base_date']),

        (CategoryByTime('base_hour'), ['base_hour']),
        (CycleSineHour('base_hour'), ['base_hour']),
        (CycleCosineHour('base_hour'), ['base_hour']),

        (GroupMeanStatistics('base_hour', 'target'), ['base_hour', 'target']),

        # (RoadLength(['start_latitude', 'start_longitude', 'end_latitude', 'end_longitude']),
        #  ['start_latitude', 'start_longitude', 'end_latitude', 'end_longitude']),

        (DistanceFromLandmark(['start_latitude', 'start_longitude']),
         ['start_latitude', 'start_longitude']),

        (HasOneNodeName(['start_node_name', 'end_node_name']),
         ['start_node_name', 'end_node_name']),

        (HasTurnRestricted(['start_turn_restricted', 'end_turn_restricted']),
         ['start_turn_restricted', 'end_turn_restricted']),

        (StringRegexRoadName('road_name'), ['road_name']),

        (CategoryAreaCluster(['start_latitude', 'start_longitude']), ['start_latitude', 'start_longitude'])
    )
    return feature

if __name__ == '__main__':

    data = read_parquet('data/road.parquet').head()

    pipeline = pipeline()

    print(pipeline.fit_transform(data))