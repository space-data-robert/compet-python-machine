import sys
import warnings
import numpy as np
import pandas as pd
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)

import sklearn.neighbors._base
sys.modules['sklearn.neighbors.base'] = sklearn.neighbors._base

from missingpy import MissForest
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer


def household_rate(data: pd.DataFrame) -> pd.DataFrame:
    household_rate: pd.Series = (
        data.total_household_cnt - data.empty_household_cnt) / data.total_household_cnt

    return household_rate.to_frame()

def around_subway_bus_cnt(data: pd.DataFrame) -> pd.DataFrame:
    around_subway_bus_cnt: pd.Series = data.around_subway_cnt + data.around_bus_cnt

    return around_subway_bus_cnt.to_frame()

def parking_per_house(data: pd.DataFrame) -> pd.DataFrame:
    parking_per_house: pd.Series = data.parking_lot_cnt + data.total_household_cnt

    return parking_per_house.to_frame()





if __name__ == '__main__':
    target_nm = 'registered_vehicle_cnt'

    train_x = pd.read_parquet('parking.parquet')
    train_y = train_x.pop(target_nm)

    feat_categories_nm = train_x.dtypes[train_x.dtypes == 'object'].index
    feat_nummeries_nm = train_x.dtypes[train_x.dtypes != 'object'].index

    feat_all_nm = list(feat_categories_nm) + list(feat_nummeries_nm)

    HouseholdRate: object = FunctionTransformer(household_rate)
    AroundSubwayBusCnt: object = FunctionTransformer(around_subway_bus_cnt)
    ParkingPerHouse: object = FunctionTransformer(parking_per_house)


    pipeline = make_pipeline(
        make_column_transformer(
            (FunctionTransformer(func=np.log1p), feat_nummeries_nm),
            (OrdinalEncoder(), feat_categories_nm),
        ),
        MissForest(random_state=27),
        make_column_transformer(
            (HouseholdRate, ['total_household_cnt', 'empty_household_cnt']),
            (AroundSubwayBusCnt, ['around_subway_cnt', 'around_bus_cnt']),
            (ParkingPerHouse, ['parking_lot_cnt', 'total_household_cnt']),
        ),
    )


