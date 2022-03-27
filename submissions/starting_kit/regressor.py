import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator
import xgboost as xgb

class Regressor(BaseEstimator):
    def __init__(self):
        
        self.model = xgb.XGBRegressor(max_depth=7, n_estimators=800, gamma=0.0, min_child_weight=1, 
                          subsample=0.9, colsample_bytree=0.6, colsample_bylevel=0.8,
                          reg_alpha=0.2, reg_lambda=0.5, objective='reg:squarederror')

    def fit(self, X, Y):
        X = X.select_dtypes(include=[np.number])
        self.X_maxs = np.max(X, axis=0)
        X = X / self.X_maxs
        self.model.fit(X, Y)
 
    def predict(self, X):
        X = X.select_dtypes(include=[np.number])

        # normalization
        X = X / self.X_maxs

        preds = self.model.predict(X)

        return preds