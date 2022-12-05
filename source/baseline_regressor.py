import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

import scipy.stats as stats

from autosklearn.regression import AutoSklearnRegressor
from autosklearn.metrics import root_mean_squared_error

from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_regression


def baseline(data_name, target_name):
    feature_count = 50
    random_state = 27
    search_minute = 5

    data = pd.read_csv(data_name)
    print(f'data shape = {data.shape}')

    x_data = data.copy()

    x_data = pd.get_dummies(
        x_data, dummy_na=True
    ).dropna(axis=0)

    y_data = x_data.pop(target_name)

    stat, pval = stats.kstest(
        y_data, 'norm',
        args=(y_data.mean(), y_data.var() ** 0.5)
    )
    print(f'y norm pvalue = {pval:.3f}')

    if pval <= 0.5:
        y_data = y_data.apply(lambda x: np.log1p(x))
        print('y apply to log func.')

    x_train, x_test, y_train, y_test = train_test_split(
        x_data, y_data, test_size=0.3, random_state=random_state
    )
    print(f'train shape = {x_train.shape}, {y_train.shape}')

    selector = SelectKBest(f_regression, k=feature_count)
    selector.fit_transform(x_train, y_train)

    selected_mask = selector.get_support()
    print(f'feature count = {selected_mask.sum()}')

    all_feature = x_train.columns
    selected_feature = all_feature[selected_mask]

    automl = AutoSklearnRegressor(
        load_models=True,
        time_left_for_this_task=60*search_minute,
        resampling_strategy='cv',
        metric=root_mean_squared_error,
        ensemble_size=5,
        memory_limit=10**4,
        seed=random_state,
    )
    automl.fit(
        x_train[selected_feature],
        y_train
    )

    y_pred = automl.predict(
        x_test[selected_feature]
    )

    test_score = root_mean_squared_error(
        y_test, y_pred
    )
    print(f'test score = {-test_score:.3f}')

    print(automl.leaderboard())

    return automl


if __name__ == '__main__':
    baseline(data_name='house_price.csv', target_name='SalePrice')
