import numpy as np
import os, sys
import math
from sklearn.linear_model import LinearRegression, Ridge, SGDRegressor, Lasso, ElasticNet
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVR
from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn import metrics
import pandas as pd
from matplotlib import pyplot as plt
from modules.preprocessing import *
import modules.statistics as st 
# from modules.learning import *
# from modules.io import *
import warnings


warnings.simplefilter(action='ignore', category=FutureWarning)


def fit_linear_model(df, feats, target, a=1e-4, deg=3, method='ridge', fit_intercept=True, include_bias=True):
    """
    Fits a regression model on a given dataframe, and returns the model, the predicted values and the associated
    scores. Applies Ridge Regression with polynomial features.

    Args:
        df: The input dataframe.
        feats: List of names of columns of df. These are the feature variables.
        target: The name of a column of df corresponding to the dependent variable.
        a: A positive float. Regularization strength parameter for the linear least squares function
        (the loss function) where regularization is given by the l2-norm.
        deg: The degree of the regression polynomial.

    Returns:
        pipeline: The regression model. This is an instance of Pipeline.
        y_pred: An array with the predicted values.
        r_sq: The coefficient of determination “R squared”.
        mae: The mean absolute error.
        me: The mean error.
        mape: The mean absolute percentage error.
        mpe: The mean percentage error.
    """
    df_x = df[feats]
    df_y = df[target]
    X = df_x.values
    y = df_y.values
    polynomial_features = PolynomialFeatures(degree=deg, include_bias=include_bias)
    if method == 'ridge':
        model = Ridge(alpha=a, fit_intercept=fit_intercept)

    elif method == 'ols':
        model = LinearRegression(fit_intercept=fit_intercept)
    elif method == 'rf':
        model = RFRegressor(n_jobs = -1)
    else:
        print('Unsupported method')


    pipeline = Pipeline([("polynomial_features", polynomial_features),
                         ("regression", model)])


    pipeline.fit(X, y)
    y_pred = pipeline.predict(X)
    r_sq, mae, me, mape, mpe, med = st.score(y, y_pred)
    return pipeline, y_pred, r_sq, mae, me, mape, mpe, med

def predict(df_test, model, feats, target):
    """
    Applies a regression model to predict values of a dependent variable for a given dataframe and
    given features.

    Args:
        df_test: The input dataframe.
        model: The regression model. Instance of Pipeline.
        feats: List of strings: each string is the name of a column of df_test.
        target: The name of the column of df corresponding to the dependent variable.
    Returns:
        y_pred: Array of predicted values.
    """

    df_x = df_test[feats]
    df_y = df_test[target]
    X = df_x.values
    y_true = df_y.values
    y_pred = model.predict(X)
    return y_pred


def train_on_reference_points(df, w_train, ref_points, feats, target, random_state=0):
    """
    Trains a regression model on a training set defined by segments of a dataframe.
    These segments are defined by a set of starting points and a parameter indicating their duration.
    In each segment, one subset of points is randomly chosen as the training set and the remaining points
    define the validation set.

    Args:
        df: Input dataframe.
        w_train: The duration, given as a number of days, of the segments where the model is trained.
        ref_points: A list containing the starting date of each segment where the model is trained.
        feats: A list of names of columns of df corresponding to the feature variables.
        target: A name of a column of df corresponding to the dependent variable.
        random_state: Seed for a random number generator, which is used in randomly selecting the validation
        set among the points in a fixed segment.

    Returns:
        model: The regression model. This is an instance of Pipeline.
        training_scores: An array containing scores for the training set. It contains the coefficient
        of determination “R squared”, the mean absolute error, the mean error, the mean absolute percentage error.
        validation_scores: An array containing scores for the validation set. It contains the coefficient
        of determination “R squared”, the mean absolute error, the mean error, the mean absolute percentage error.
    """
    df_train = pd.DataFrame([])
    df_val = pd.DataFrame([])
    
    for idx in range(ref_points.size):
        
        
        d_train_stop = pd.to_datetime(ref_points[idx]) + pd.Timedelta(days=w_train)
        df_tmp = df.loc[ref_points[idx]:str(d_train_stop)]
        df_tmp2 = df_tmp.sample(frac=1, random_state=random_state)
        # added random state for reproducibility during experiments
        size_train = int(len(df_tmp2) * 0.80)
        # df_train = df_train.append(df_tmp2[:size_train])
        # df_val = df_val.append(df_tmp2[size_train:])
        df_train = pd.concat([df_train, df_tmp2[:size_train]], ignore_index=True)
        df_val = pd.concat([df_val, df_tmp2[size_train:]], ignore_index=True)

        

    model, y_pred_train, r_sq_train, mae_train, me_train, mape_train, mpe_train, Me_train = fit_linear_model(df_train, 
                                                                                                             feats, target)
    y_pred_val = predict(df_val, model, feats, target)
    r_sq_val, mae_val, me_val, mape_val, mpe_val, Me_val = st.score(df_val[target].values, y_pred_val)
    training_scores = np.array([r_sq_train, mae_train, me_train, mape_train, Me_train])
    validation_scores = np.array([r_sq_val, mae_val, me_val, mape_val, mpe_val, Me_val])

    print('Training Metrics:')
    print(f'MAE:{training_scores[1]:.3f} \nME(true-pred):{training_scores[2]:.3f} \nMAPE:{training_scores[3]:.3f} \nR2: {training_scores[0]:.3f}\n')
    print('Validation Metrics:')
    print(f'MAE:{validation_scores[1]:.3f} \nME(true-pred):{validation_scores[2]:.3f} \nMAPE:{validation_scores[3]:.3f} \nMPE:{validation_scores[4]:.3f} \nR2: {validation_scores[0]:.3f}\n')
    return model, training_scores, validation_scores


def calculate_pi(weeks_train, start_date, end_date, path, dataset_id,
                cp_starts, cp_ends, query_modelar=False):

    filename = path

#     res_df = load_power_index_cql(start_date=start_date, end_date='end_date',
#                                   dataset=dataset_id, cp_starts=cp_starts, cp_ends=cp_ends,
#                                   weeks_train=weeks_train, query_modelar=query_modelar)
#     print(res_df.shape())
#     if res_df is not None:
#         # if power index is stored in cassandra dont recalculate
#         res_df.set_index('timestamp', inplace=True)
#         res_df.index = pd.DatetimeIndex(res_df.index)
#         res_df.columns = ['power_index', 'estimated_power_lost']
#         res_df.sort_index(inplace=True)
#         return res_df



#     if not query_modelar:
#         df = pd.read_csv(filename, index_col = 'timestamp')
#     else:
#         print("Use modelar")
#         df = load_df_modelar([1, 2, 3], ['irradiance', 'power', 'mod_temp'],
#                              hostname='localhost', limit=10**6)
    df = pd.read_csv(filename, index_col = 'timestamp')
    df = df.dropna()
    df.index = pd.DatetimeIndex(df.index)
    feats = ['irradiance', 'mod_temp']
    target = 'power'
    w_train = weeks_train * 1
    
    if start_date is None:
        start_date = df.index.min()
    if end_date is None:
        end_date = df.index.max()

    df = filter_dates(df, start_date, end_date)

    scaler = MinMaxScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns, index=df.index)
    ref_points = pd.Index(pd.Series(cp_ends))
    ref_points = pd.Index(['2013-09-23', '2013-11-20', '2013-09-19', '2013-01-13',
               '2013-11-06', '2013-06-26', '2013-09-26', '2013-08-22',
               '2013-03-23'])

    model1, _, _ = train_on_reference_points(df_scaled, w_train, ref_points, feats, target)

    est = predict(df_scaled, model1, feats, target)
    df_est = pd.DataFrame(est, columns=['estimated_power'], index=df_scaled.index)
    pi_daily = (df_scaled.power/df_est.estimated_power).resample("1D").median()
    pi_daily = np.clip(pi_daily, 0, 1)
    pi_daily = pi_daily.ffill()
    df_daily = df.resample("1D").sum()
    df_daily = df_daily.ffill()
    daily_loss = calculate_daily_loss(pi_daily, df_daily)
    df_result = pd.concat([pi_daily, daily_loss], axis=1)
    df_result.columns = ['power_index', 'estimated_power_lost']
    print(df_result)
    return df_result


def calculate_daily_loss(pi_daily, df_daily):
    daily_loss = (df_daily.power/pi_daily) - df_daily.power
    return daily_loss


