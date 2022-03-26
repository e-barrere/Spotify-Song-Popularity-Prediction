import os
import pandas as pd
import rampwf as rw

from sklearn.model_selection import ShuffleSplit
import numpy as np
from sklearn.metrics import mean_squared_error

from rampwf.score_types.base import BaseScoreType




quick_mode = os.getenv('RAMP_TEST_MODE', 0)

if quick_mode:
    _train = 'train_small.csv'
    _test = 'test_small.csv'
else:
    _train = 'train.csv'
    _test = 'test.csv'

problem_title = 'Spotify song popularity prediction'
_target_column_names = ['popularity']

# A type (class) which will be used to create wrapper objects for y_pred
Predictions = rw.prediction_types.make_regression(
    label_names=_target_column_names
)

# An object implementing the workflow
workflow = rw.workflows.Regressor()


class RMSE(BaseScoreType):
    is_lower_the_better = True
    minimum = 0.0
    maximum = float('inf')

    def __init__(self, name='RMSE', precision=2):
        self.name = name
        self.precision = precision

    def __call__(self, y_true, y_pred):
        return np.sqrt(mean_squared_error(y_true, y_pred))

score_types = [
    RMSE()
]


def _read_data(path, f_name):
    data = pd.read_csv(os.path.join(path, 'data', f_name))

    labels = data[_target_column_names].to_numpy()

    features = data.iloc[:, 10:20].to_numpy()

    return features, labels


def get_train_data(path='.'):
    f_name = _train
    return _read_data(path, f_name)


def get_test_data(path='.'):
    f_name = _test
    return _read_data(path, f_name)


def get_cv(X, y):
    cv = ShuffleSplit(n_splits=3, test_size=0.2, random_state=42)
    return cv.split(X, y)
