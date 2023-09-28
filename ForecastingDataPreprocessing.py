import os, sys
import warnings
warnings.filterwarnings('ignore')
module_path = os.path.abspath(os.path.join('../../'))
if module_path not in sys.path:
    sys.path.append(module_path)
print(sys.path)
import lightgbm
from sklearn.preprocessing import MinMaxScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from feature_engine.timeseries.forecasting import LagFeatures, WindowFeatures
from feature_engine.datetime import DatetimeFeatures
from copy import deepcopy
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error 
from sklearn.metrics import mean_absolute_percentage_error as mape
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.multioutput import MultiOutputRegressor, RegressorChain
from sklearn.pipeline import Pipeline 



class DataPreprocessor:
    def __init__(self):
        self.df = pd.DataFrame()
        self.train_index = pd.Index([])
        self.test_index = pd.Index([])
        self.scaler = MinMaxScaler(feature_range=(0, 1))
    
 
    def load_data(self, file_path):
        self.df = pd.read_csv(file_path, index_col=0)
        self.df.index = pd.to_datetime(self.df.index)
    
    def filter_dates(self, start_date, end_date):
        self.df = self.df.loc[start_date:end_date]

    def set_train_test_index(self, train_date, test_date):
        self.train_index = self.df.loc[:train_date].index
        self.test_index = self.df.loc[test_date:].index

    def remove_columns(self, columns):
        self.df = self.df[[col for col in self.df.columns if col not in columns]]

    def scale_data(self):
        turblabel=self.df['label']
        self.scaler.fit(self.df.loc[self.train_index])
        self.df = pd.DataFrame(self.scaler.transform(self.df), columns=self.df.columns, index=self.df.index)
        self.df['label'] = turblabel
        
#     def outliers_IQR(df):
#         Q1 = df.quantile(0.10)
#         Q3 = df.quantile(0.90)
#         IQR = Q3 - Q1
#         df_iqr = df[~((df < (Q1 - 1.5 * IQR)) | (df >(Q3 + 1.5 * IQR))).any(axis=1)]
#         return df_iqr 
    
    def clean_outliers(self):
        df_outliers = self.df.copy()
        for l in self.df['label'].unique():
            l = int(l)
            df_label = self.df[self.df['label'] == l].copy()

            df_label.loc[(df_label.index.isin(self.train_index)) &
                         ((df_label['Grd_Prod_Pwr_avg'] < 0.02) & (df_label['Amb_WindSpeed_avg'] > 0.13)),
                         ['Grd_Prod_Pwr_avg', 'Grd_Prod_Pwr_min',
                          'Grd_Prod_Pwr_max', 'Grd_Prod_Pwr_std']] = np.nan

#             df_iqr = outliers_IQR(df_label.loc[df_label.index.isin(self.train_index)][['Grd_Prod_Pwr_avg', 'Amb_WindSpeed_avg']])
            temp=df_label.loc[df_label.index.isin(self.train_index)][['Grd_Prod_Pwr_avg', 'Amb_WindSpeed_avg']]
            Q1 = temp.quantile(0.10)
            Q3 = temp.quantile(0.90)
            IQR = Q3 - Q1
            df_iqr = temp[~((temp < (Q1 - 1.5 * IQR)) | (temp >(Q3 + 1.5 * IQR))).any(axis=1)]
    
            df_label.loc[(~df_label.index.isin(df_iqr.index)) & (df_label.index.isin(self.train_index)),
                          ['Grd_Prod_Pwr_avg', 'Grd_Prod_Pwr_min', 'Grd_Prod_Pwr_max', 'Grd_Prod_Pwr_std']] = np.nan

            df_label = np.clip(df_label, 0, 1)
            df_label['label'] = l

            self.df.loc[self.df['label'] == l] = df_label

    def impute_missing_values(self):
        imp_mean = IterativeImputer(random_state=42, max_iter=50, skip_complete=True, tol=1e-2,
                                    verbose=0, imputation_order='ascending', initial_strategy='mean',
                                    n_nearest_features=None)

        for l in self.df['label'].unique():
            l = int(l)
            df_label = self.df[self.df['label'] == l].copy()
            imp_mean.fit(df_label.loc[df_label.index.isin(self.train_index)].values)
            df_label = pd.DataFrame(np.clip(imp_mean.transform(df_label.values), 0, 1),
                                    columns=df_label.columns, index=df_label.index)
            df_label['label'] = l
            self.df.loc[self.df['label'] == l] = df_label

    def create_lagged_features(self, periods):
        df_list = []
        for l in self.df['label'].unique():
            lag_f = LagFeatures(periods=periods)
            df_temp = lag_f.fit_transform(self.df.loc[self.df['label'] == l])
            df_temp = df_temp.dropna()

            dtfs = DatetimeFeatures(
                variables="index",
                features_to_extract=["month", "hour", "day_of_week"],
                drop_original=False
            )

            df_temp = dtfs.fit_transform(self.df.loc[self.df['label'] == l])
            df_list.append(df_temp.copy())

        self.df = pd.concat(df_list, axis=0)

    def create_future_features(self, future_steps):
        df_list = []
        for l in self.df['label'].unique():
            df_list_inner = [self.df.loc[self.df['label'] == l]]
            df_future = self.df.loc[self.df['label'] == l, ['Grd_Prod_Pwr_min', 'MeanWindSpeedUID_10.0m',
                                                            'MeanWindSpeedUID_100.0m','DirectionUID_10.0m', 
                                                            'DirectionUID_100.0m', "month", "hour", "day_of_week"]]
            for i in range(1, future_steps+1):
                if i != 0:
                    df_temp = df_future.shift(-i)
                    new_columns = []  # Create an empty list to store the new column names

                    for c in df_future.columns:
                        new_column = c + "_(t+" + str(i) + ")"  # Append the time indicator to the column name
                        new_columns.append(new_column)  # Add the new column name to the list

                    df_temp.columns = new_columns  # Assign the list of new column names to the columns of df_temp

                    df_list_inner.append(df_temp)

            df_f = pd.concat(df_list_inner, axis=1)
            df_f = df_f.dropna()
            df_list.append(df_f)

        self.df = pd.concat(df_list, axis=0)

    def split_train_test(self):
        self.df_train = self.df.loc[self.df.index.isin(self.train_index)]
        self.df_test = self.df.loc[self.df.index.isin(self.test_index)]
        
        
        
        
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
            
         
            
            
if __name__ == '__main__':
    
    
    preprocessor = DataPreprocessor()
    print('ok')
    preprocessor.load_data("/home/pgidarakos/SHADOWJER/full_park_data_labels_vs2.csv")
    print('ok')

    preprocessor.filter_dates("2021-07-01", "2022-08-01")
    print('ok')

    preprocessor.set_train_test_index("2022-07-01", "2022-07-01")
    preprocessor.remove_columns(['Rtr'])
    preprocessor.scale_data()
    print('ok')
    preprocessor.clean_outliers()
    preprocessor.impute_missing_values()
    preprocessor.create_lagged_features([1, 2, 3])
    preprocessor.create_future_features(48)
    preprocessor.split_train_test()
    df_train = preprocessor.df_train
    print(df_train.shape)
    df_test = preprocessor.df_test
    print(df_test.shape)

    df_f=preprocessor.df
    print(df_f.shape)
    lgbmr = LGBMRegressor(n_jobs=-1, deterministic=True)
    mod = RegressorChain(lgbmr)
    hyper_params={}
    hyper_params['lgbm_regression'] = {
        'regression__base_estimator__learning_rate': 0.05,
        'regression__base_estimator__max_bin': 255,
        'regression__base_estimator__min_child_samples': 10,
        'regression__base_estimator__subsample': 0.5,
        'regression__base_estimator__subsample_freq': 1,
        'regression__base_estimator__colsample_bytree': 0.5,
        'regression__base_estimator__n_estimators': 500,
        'regression__base_estimator__max_depth': 10
    }
    regression_models={}
    pipeline = Pipeline([("regression", mod)])
    regression_models['lgbm_regression']= (pipeline, hyper_params)
    

    target_features = [x for x in df_f.columns if 'Grd_Prod_Pwr_min_(t+' in x]
    fit_features_test = [x for x in df_f.columns if 'Grd_Prod_Pwr_min_(t+' not in x]
    fit_features = fit_features_test

    # Initialize your dataframes and other necessary variables
    df_train = df_train
    df_test = df_test
    regression_models = regression_models
    hyper_params = hyper_params
    fit_features = fit_features
    target_features = target_features

    # Create an instance of the RegressionModel class
    model = RegressionModel(df_train, df_test, regression_models, hyper_params, fit_features, target_features)
    print('modelok')

    # Run the instance
    model.run_instance()
