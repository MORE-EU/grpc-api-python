import sys
sys.path.insert(0,'../')
from modules.preprocessing import *
from modules.io import *
from modules.learning import *
from modules.patterns import *
from modules.statistics import *
from modules.plots import *
from matplotlib import pyplot as plt
from timeit import default_timer as timer
plt.style.use('ggplot')
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer
from tqdm.notebook import tqdm
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from feature_engine.datetime import DatetimeFeatures
from lightgbm import LGBMRegressor
from sklearn.multioutput import MultiOutputRegressor, RegressorChain
import lightgbm
from copy import deepcopy
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_absolute_percentage_error as mape

def mape1(y_true, y_pred):

    """
    Computes the Mean Absolute Percentage Error between the 2 given time series

    Args:
        y_true: A numpy array that contains the actual values of the time series.
        y_pred: A numpy array that contains the predicted values of the time series.

    Return:
        Mean Absolute Percentage Error value.


    """
    if y_true.ndim >= 2 and y_pred.ndim >= 2:
        mapes = []
#         nom = np.mean(np.abs(y_true - y_pred), axis=1)
#         denom = np.mean(y_true, axis=1)
#         mapes = nom/denom
#         mape1 = np.mean(mapes)
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

def percentage_of_misses(true, predicted):
    ratio = np.sum(true < predicted) / len(true)
    return ratio

def naive_baseline(df_test):
       
    preds = []
    for i in range(df_test.shape[0]):
        t = [x for x in df_test.columns if 'Grd_Prod_Pwr_min_(t+' in x]
        pred = np.zeros(len(t))
        pred =  pred + df_test.iloc[i, :]['Grd_Prod_Pwr_min']
        preds.append(pred)
    preds = np.vstack(preds)
    return preds

def estimate(path='', start_date = "2021-07-01",
    end_date= "2022-08-01", lags=3, future_steps=48,
                 query_modelar=False, dataset_id='beico'):
    path = f'/home/pgidarakos/Forecasting_30min/data1h.csv'
    if start_date is None:
        start_date = '2021-07-01'
    if end_date is None:
        end_date = '2022-08-01'
    df = pd.read_csv(path, index_col=0)
    df.index = pd.to_datetime(df.index)

    # Use data after June of 2021 as instructed
    df = filter_dates(df, "2021-07-01", "2022-08-01")

    train_index, test_index = df.loc[:"2022-07-01"].index, df.loc["2022-07-01":].index
    df = df[[col for col in df.columns if 'Rtr' not in col]]
    scaler = MinMaxScaler(feature_range=(0,1))
    scaler.fit(df.loc[train_index])
    turbine_labels = df['label']


    df = pd.DataFrame(scaler.transform(df), columns=df.columns, index=df.index)
    df['label'] = turbine_labels
    df_outliers=df.copy()
    for l in np.unique(df.label):
        l = int(l)
        df_label = df[df.label==l].copy()


        df_iqr = outliers_IQR(df_label.loc[df_label.index.isin(train_index)][['Grd_Prod_Pwr_avg', 'Amb_WindSpeed_avg']])
        df_label.loc[(~df_label.index.isin(df_iqr.index)) 
                     & (df_label.index.isin(train_index)),
                      ['Grd_Prod_Pwr_avg', 'Grd_Prod_Pwr_min',
                       'Grd_Prod_Pwr_max', 'Grd_Prod_Pwr_std']]=np.nan

        df_label = np.clip(df_label, 0, 1)
        df_label['label'] = l
        df.loc[df.label==l]=df_label
        
    imp_mean = IterativeImputer(random_state=42, max_iter=50, skip_complete=True, tol=1e-2, 
                            verbose=0, imputation_order='ascending', initial_strategy='mean',
                            n_nearest_features=None)

    for l in np.unique(df.label):
        l = int(l)
        df_label = df[df.label==l].copy()
        imp_mean.fit(df_label.loc[df_label.index.isin(train_index)].values)
        df_label = pd.DataFrame(np.clip(imp_mean.transform(df_label.values), 0, 1),
                                columns=df_label.columns, index=df_label.index)
        df_label.label = l
        df.loc[df.label==l]=df_label
        
    df_stops = pd.read_csv('/home/pgidarakos/Forecasting_30min/stops_per_turbine_1h.csv', index_col=0, converters={'stop_times': pd.eval})
    df_stops.index = df_stops.index.map(lambda x: int(x.replace('WT', "")))
    df_stops

    df_list = []

    for l in np.unique(df.label):
        df_temp = df[df.label==l].copy()
        stop_index = pd.Index(df_stops.loc[l, 'stop_times'])
        df_temp.loc[df_temp.index.isin(stop_index), 'Grd_Prod_Pwr_min'] = 0
    #     df_temp.loc[df_temp.index.isin(stop_index.intersection(train_index)), 'Grd_Prod_Pwr_min'] = np.nan # only in test set
        df_list.append(df_temp)

    df = pd.concat(df_list, axis=0)

    lags = 3
    df_list = []

    for l in np.unique(df.label):
    #     lag_f = LagFeatures(periods=[1, 2, 3])
    #     df_temp = lag_f.fit_transform(df.loc[df.label==l])

    #     df_temp = df_temp.dropna()
        df_lags_list = [df.loc[df.label==l]]
        df_temp = df.loc[df.label==l]
        for i in range(1, lags+1):
            df_temp = df_temp.shift(i)
            df_temp.columns = [c+f"_lag_{i}" for c in df_temp.columns]
            df_lags_list.append(df_temp.copy())

        df_lagged = pd.concat(df_lags_list, axis=1)

        dtfs = DatetimeFeatures(
            variables="index",
            features_to_extract=["month", "hour", "day_of_week"],
            drop_original=False
        )

        df_temp = dtfs.fit_transform(df_lagged.loc[df_lagged.label==l])
        df_list.append(df_temp.copy())

    df = pd.concat(df_list, axis=0)


    future_steps = 48
    df_list = []
    for l in np.unique(df.label):
        df_list_inner = [df.loc[df.label==l]]
        df_future = df.loc[df.label==l, ['Grd_Prod_Pwr_min', 'MeanWindSpeedUID_10.0m', 'MeanWindSpeedUID_100.0m', 
                                         'DirectionUID_100.0m', 'DirectionUID_100.0m', "month", "hour", "day_of_week"]]
        for i in range(1, future_steps+1):
            if i != 0:
                df_temp = df_future.shift(-i)
                df_temp.columns = [c+f"_(t+{i})" for c in df_future.columns]
                df_list_inner.append(df_temp) 

        df_f = pd.concat(df_list_inner, axis=1)
        df_f.head()
        df_list.append(df_f)

    df_f = pd.concat(df_list, axis=0)
    print(df_f.shape)
    df_f = df_f.dropna()
    print(df_f.shape)
    df_train, df_test = df_f.loc[df_f.index.isin(train_index)], df_f.loc[df_f.index.isin(test_index)]
    

    regression_models = {}

    lgbmr = LGBMRegressor(n_jobs=32, deterministic=True)
    model = RegressorChain(lgbmr)
    parameters = {'regression__base_estimator__learning_rate': list(np.linspace(0.01, 0.3, 10)),
                  'regression__base_estimator__extra_trees': [True, False],
                  'regression__base_estimator__reg_alpha': [10**x for x in range(-6, 3, 1)],
                  'regression__base_estimator__reg_lambda': [10**x for x in range(-6, 3, 1)],
                  'regression__base_estimator__max_bin': [31, 63, 127, 255],
                  'regression__base_estimator__min_child_samples': [5, 10, 20],
                  'regression__base_estimator__num_leaves': [7, 15, 31],
                  'regression__base_estimator__subsample': [0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
                  'regression__base_estimator__subsample_freq': [1, 5, 10, 20],
                  'regression__base_estimator__colsample_bytree': [0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
                  'regression__base_estimator__n_estimators': list(range(100, 1000, 100)),
                  'regression__base_estimator__max_depth': list(range(1, 11, 2))}

    pipeline = Pipeline([("regression", model)])
    regression_models['lgbm_regression'] = (pipeline, parameters)

    target_features = [x for x in df_f.columns if 'Grd_Prod_Pwr_min_(t+' in x]
    fit_features_test = [x for x in df_f.columns if 'Grd_Prod_Pwr_min_(t+' not in x]
    fit_features = fit_features_test

    print("Training set")
    print(df_train.shape[0])
    print(len(list(target_features)))
    print(len(list(fit_features_test)))
    ts_cv = TimeSeriesSplit(n_splits=5,)
    scorer = make_scorer(mape1, greater_is_better=False)
    hyper_params = {}
    # times = {}
    # for key, (pipeline, params) in regression_models.items():
    #     print(f"Tuning and validating {key}...")
    #     print(params.keys())
    #     print(pipeline)
    #     start = timer()
    #     temp = perform_grid_search(df_train, fit_features_test, target_features, scorer, pipeline, params, randomized=True, cv=ts_cv, n_jobs=2, n_iter=250)
    #     end = timer()
    #     times[key] = end-start
    #     print(f'Selected hyper-parameters for {key} \n {temp}')
    #     hyper_params[key] = temp
    # print(times)
    hyper_params['lgbm_regression'] = {'regression__base_estimator__learning_rate': 0.05,
                                       'regression__base_estimator__max_bin': 255,
                                       'regression__base_estimator__min_child_samples': 10, 
                                       'regression__base_estimator__subsample': 0.9,
                                       'regression__base_estimator__subsample_freq': 1, 
                                       'regression__base_estimator__colsample_bytree': 0.9, 
                                       'regression__base_estimator__n_estimators': 500, 
                                       'regression__base_estimator__max_depth': 10}
    print("Fit with selected parameters and predict on test set")
    df_test = df_test
    mapes = {}
    maes = {}
    rmses = {}
    for key, (pipeline, _) in regression_models.items():
        chosen_params = hyper_params[key]

        history = df_train
        result = []
        result_l = []
        result_u = []
        train_mapes = []
        print(history.shape)
        temp_pipeline = deepcopy(pipeline)
        temp_pipeline1 = deepcopy(pipeline)
        quantile_params = deepcopy(chosen_params)
        pipeline, y_pred, r_sq, mae_, me, mape_train, mpe = fit_pipeline(history, fit_features, 
                                                                         target_features, pipeline, chosen_params)
        quantile_params['regression__base_estimator__objective'] = 'quantile'
        quantile_params['regression__base_estimator__alpha']= 0.1
        pipeline_lower, _, _, _, _, _, _ = fit_pipeline(history, fit_features, target_features, temp_pipeline1, quantile_params)
        print("Training Mape")
        print(mape_train)
    all_mapes = []
    p_list_pred = []
    p_list_pred_lower = []
    p_list = []
    for l in np.unique(df.label):
        try:
            unq_idx = test_index.drop_duplicates()
            temp = np.empty((unq_idx.shape[0], future_steps))
            temp[:] = 1e-6
            result_container = pd.DataFrame(temp.copy(), index=unq_idx)
            y_pred_test = np.clip(pipeline.predict(df_test.loc[df_test.label==l][fit_features].values), 0, 1)
            result_temp = pd.DataFrame(y_pred_test, index=df_test.loc[df_test.label==l].index)
            result_container.loc[result_temp.index] = result_temp

            result_container_lower = pd.DataFrame(temp.copy(), index=unq_idx)
            y_pred_test_lower = np.clip(pipeline_lower.predict(df_test.loc[df_test.label==l][fit_features].values), 0, 1)
            result_temp_lower = pd.DataFrame(y_pred_test_lower, index=df_test.loc[df_test.label==l].index)
            result_container_lower.loc[result_temp_lower.index] = result_temp_lower



            gt_container = pd.DataFrame(temp.copy(), index=unq_idx)
            y_test = df_test.loc[df_test.label==l][target_features]
            gt_temp = pd.DataFrame(y_test, index=df_test.loc[df_test.label==l].index)
            gt_container.loc[gt_temp.index] = gt_temp
            p_list_pred.append(result_container.copy())
            p_list_pred_lower.append(result_container_lower.copy())
            p_list.append(gt_container.copy())
            all_mapes.append(mape1(y_test, y_pred_test))
        except:
            print(l)
            
    park_mat_pred = np.dstack(p_list_pred)
    park_mat_pred_lower = np.dstack(p_list_pred_lower)
    park_mat = np.dstack(p_list)
    avg_test_preds = np.sum(park_mat_pred, axis=2)
    avg_test_preds_lower = np.sum(park_mat_pred_lower, axis=2)
    avg_test = np.sum(park_mat, axis=2)
    print('Mape on aggregate test set')
    print(mape1(avg_test.ravel(), avg_test_preds.ravel()))
    mapes['lgbm'] = mape1(avg_test.ravel(), avg_test_preds.ravel())
    mae_l = mae(avg_test, avg_test_preds)
    rmse_l = np.sqrt(mse(avg_test, avg_test_preds))
    maes['lgbm'] = mae_l
    rmses['lgbm'] = rmse_l
    
    print("Model metrics on test set")
    print("MAPE", mape1(avg_test.ravel(), avg_test_preds.ravel()))
    print("MAE", mae_l)
    print("RMSE", rmse_l)
    
    
    p_baseline_pred = []
    for l in np.unique(df.label):
        try:
            unq_idx = test_index.drop_duplicates()
            temp = np.empty((unq_idx.shape[0], future_steps))
            temp[:] = 0
            baseline_container = pd.DataFrame(temp.copy(), index=unq_idx)
            p_baseline = naive_baseline(df_test[df_test.label==l])
            baseline_temp = pd.DataFrame(p_baseline, index=df_test.loc[df_test.label==l].index)
            baseline_container.loc[baseline_temp.index] = baseline_temp
            p_baseline_pred.append(baseline_container)
        except:
            print(l)

    park_mat_base_pred = np.dstack(p_baseline_pred)
    avg_base_preds = np.sum(park_mat_base_pred, axis=2)

    m = mape1(avg_test.ravel(), avg_base_preds.ravel())
    mae_b = mae(avg_test, avg_base_preds)
    rmse_b = np.sqrt(mse(avg_test, avg_base_preds))
    mapes['baseline'] = m
    maes['baseline'] = mae_b
    rmses['baseline'] = rmse_b
    print("Baseline metrics on test set")
    print("MAPE", m)
    print("MAE", mae_b)
    print("RMSE", rmse_b)
    pom = percentage_of_misses(avg_test.ravel(), avg_test_preds_lower.ravel())
    print(f"Percentage of missses: {pom}")
    result=pd.DataFrame(y_pred_test_lower)

    column_names = [f"Grd_Prod_Pwr_min(t+{i+1})" for i in range(48)]
    result.columns = column_names

    return result



