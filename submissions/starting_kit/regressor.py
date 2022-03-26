import pandas as pd 
import numpy as np
 
from sklearn.base import BaseEstimator

import pandas as pd
import numpy as np
 

from sklearn import linear_model

class Regressor(BaseEstimator):
    def __init__(self):
        
        self.model = linear_model.LinearRegression()

    def fit(self, X, Y):

        self.X_maxs = np.max(X, axis=0)
        X = X / self.X_maxs
        self.model.fit(X, Y)
 
    def predict(self, X):

        # normalization
        X = X / self.X_maxs

        preds = self.model.predict(X)

        return preds
