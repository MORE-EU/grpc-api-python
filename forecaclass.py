

import lightgbm
from copy import deepcopy
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error 
from sklearn.metrics import mean_absolute_percentage_error as mape
import numpy as np
import pandas as pd

class RegressionModel:
    def __init__(self, df_train, df_test, regression_models, hyper_params,fit_features,target_features):
        self.df_train = df_train
        self.df_test = df_test
        self.target_features=target_features
        self.fit_features= fit_features
        self.regression_models = regression_models
        self.hyper_params = hyper_params
        self.mapes = {}
        self.maes = {}
        self.rmses = {}
        
        
    def mape1(y_true, y_pred):


        if y_true.ndim >= 2 and y_pred.ndim >= 2:
            mapes = []

            nom = np.sum(np.abs(y_true - y_pred), axis=1)
            denom = np.sum(y_true + y_pred, axis=1)
            mapes = nom/denom
            mape1 = np.mean(mapes)
        else:
            y_true = y_true.ravel()
            y_pred = y_pred.ravel()
    #         mape1 = (np.mean(np.abs(y_true-y_pred)/np.mean(y_true)))
            mape1 = (np.sum(np.abs(y_true-y_pred)/np.sum(y_true+y_pred)))

        return mape1


    def mpe1(y_true, y_pred):

  
        mpe1 = (np.mean(y_true-y_pred)/np.mean(y_true))
        return mpe1
    def score(y_true, y_pred):
        r_sq = r2_score(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        me = np.mean(y_true-y_pred)
        
        
        mpe = mpe1(y_true, y_pred)
        med = np.median(y_true-y_pred)

        return r_sq, mae, me, mape, mpe, med
    
    def fit_pipeline(self, df, feats, target, pipeline, params):
        df_x = df[feats]
        df_y = df[target]
        X = df_x.values
        y = df_y.values
        y_true=y

        pipeline.set_params(**params)
        pipeline.fit(X, y)
        y_pred = pipeline.predict(X)
        r_sq = r2_score(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        me = np.mean(y_true-y_pred)
        
        if y_true.ndim >= 2 and y_pred.ndim >= 2:
            mapes = []

            nom = np.sum(np.abs(y_true - y_pred), axis=1)
            denom = np.sum(y_true + y_pred, axis=1)
            mapes = nom/denom
            mape1 = np.mean(mapes)
        else:
            y_true = y_true.ravel()
            y_pred = y_pred.ravel()
    #         mape1 = (np.mean(np.abs(y_true-y_pred)/np.mean(y_true)))
            mape1 = (np.sum(np.abs(y_true-y_pred)/np.sum(y_true+y_pred)))
        mape=mape1
        mpe = (np.mean(y_true-y_pred)/np.mean(y_true))

        med = np.median(y_true-y_pred)
#         r_sq, mae, me, mape, mpe, _ = score(y, y_pred)
        return pipeline, y_pred, r_sq, mae, me, mape, mpe

    
    def run_instance(self):
        for key, (pipeline, _) in self.regression_models.items():
            chosen_params = self.hyper_params[key]

            history = self.df_train
            result = []
            result_l = []
            result_u = []
            train_mapes = []
            
            print(history.shape)
            
            temp_pipeline = deepcopy(pipeline)
            temp_pipeline1 = deepcopy(pipeline)
            quantile_params = deepcopy(chosen_params)
            
            pipeline, y_pred, r_sq, mae_, me, mape_train, mpe = self.fit_pipeline(history, self.fit_features, self.target_features, pipeline, chosen_params)
            
            quantile_params['regression__base_estimator__objective'] = 'quantile'
            quantile_params['regression__base_estimator__alpha'] = 0.1
            
            pipeline_lower, _, _, _, _, _, _ = self.fit_pipeline(history, self.fit_features, self.target_features, temp_pipeline1, quantile_params)
            
            print(pipeline == pipeline_lower)
            print("Training Mape")
            print(mape_train)

