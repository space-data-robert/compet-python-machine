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


if __name__ == '__main__':

    data: pd.DataFrame = pd.read_parquet('data/parking.parquet')
    print(f'data.shape = {data.shape}')

    target_name: str = 'RegisteredVehicleCount'

    x_train = data.copy()
    y_train = x_train.pop(target_name)


