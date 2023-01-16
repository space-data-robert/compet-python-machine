import sys
import warnings
import numpy as np
import pandas as pd
warnings.filterwarnings('ignore')

import sklearn.neighbors._base
sys.modules['sklearn.neighbors.base'] = sklearn.neighbors._base

from missingpy import MissForest
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder

from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import PowerTransformer

from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer

def rename_columns(data: pd.Series) -> pd.DataFrame:
    data: pd.DataFrame = pd.DataFrame(
        data, columns=(list(feat_nummeries_nm) + list(feat_categories_nm))
    )
    data[feat_categories_nm] = data[feat_categories_nm].astype('category')

    return data

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

def pipeline(train_x, train_y):
    pipeline_target: object = make_pipeline(
        FunctionTransformer(func=np.log1p)
    )
    caret_y: pd.Series = pipeline_target.fit_transform(train_y)

    print(f'Shape of pre-processed target = {caret_y.shape}')

    pipeline_feat: object = make_pipeline(
        make_column_transformer(
            (PowerTransformer(), feat_nummeries_nm),
            (OrdinalEncoder(), feat_categories_nm),
        ),
        MissForest(random_state=27),
        FunctionTransformer(func=rename_columns),

        make_column_transformer(
            (FunctionTransformer(func=None), feat_nummeries_nm),
            (OneHotEncoder(
                categories='auto', sparse=False, handle_unknown='ignore'), feat_categories_nm),
            (FunctionTransformer(
                func=household_rate), ['total_household_cnt', 'empty_household_cnt']),
            (FunctionTransformer(
                func=around_subway_bus_cnt), ['around_subway_cnt', 'around_bus_cnt']),
            (FunctionTransformer(
                func=parking_per_house), ['parking_lot_cnt', 'total_household_cnt']),
        )
    )

        caret_x: np.array = pipeline_feat.fit_transform(train_x)
        print(f'Shape of pre-processed features = {caret_x.shape}')

    return -1


if __name__ == '__main__':
    target_nm: str = 'registered_vehicle_cnt'

    train_x: pd.DataFrame = pd.read_parquet('parking.parquet')
    train_y: pd.DataFrame = train_x.pop(target_nm)

    feat_categories_nm: str = train_x.dtypes[train_x.dtypes == 'object'].index
    feat_nummeries_nm: str = train_x.dtypes[train_x.dtypes != 'object'].index

    pipeline(train_x, train_y)