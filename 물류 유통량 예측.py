import warnings
import numpy as np
import pandas as pd
warnings.filterwarnings('ignore')

from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer

import catboost as cat
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import cross_val_score

from source.base import *
from source.logistic import *


def pipeline(x_data: pd.DataFrame, y_data: pd.DataFrame) -> object:
    preprocess: object = make_column_transformer(
        (SplitSPGNumber('REC_SPG_INNB'), ['REC_SPG_INNB']),
        (SplitSPGNumber('SEND_SPG_INNB'), ['SEND_SPG_INNB']),
        (TransformedCategory(['DLGD_CLS_NM']), ['DLGD_CLS_NM'])
    )

    encoder: object = OneHotEncoder(
        categories='auto',
        sparse=False,
        handle_unknown='ignore'
    )

    model: object = cat.CatBoostRegressor(
        task_type='GPU',
        learning_rate=0.018272261776066247,
        bagging_temperature=63.512210106407046,
        n_estimators=3794,
        max_depth=11,
        random_strength=27,
        l2_leaf_reg=1.7519275289243016e-06,
        min_child_samples=88,
        max_bin=380,
        od_type='IncToDec',
        random_state=27,
        verbose=False,
    )

    pipeline: object = make_pipeline(
        preprocess,
        encoder,
        model
    )

    score: list = cross_val_score(
        pipeline,
        x_data, y_data,
        cv=RepeatedKFold(n_splits=10, n_repeats=2, random_state=27),
        verbose=10,
        scoring='neg_mean_absolute_error',
    )
    print(f'valid score = {-np.mean(score): .5f}')

    pipeline.fit(x_data, y_data)

    return pipeline


if __name__ == '__main__':

    data: pd.DataFrame = pd.read_parquet('data/logistic.parquet')
    print(f'data.shape = {data.shape}')

    target_name: str = 'INVC_CONT'

    x_data: pd.DataFrame = data.copy().head(10000)
    y_data: pd.DataFrame = x_data.pop(target_name)

    power: object = CustomPowerTransformer()

    y_data_power: np.array = power.fit_transform(y_data)

    pipeline: object = pipeline(x_data, y_data)

    y_pred_power: np.array = pipeline.predict(x_data)
    y_pred: np.array = power.inverse_transform(y_pred_power)
