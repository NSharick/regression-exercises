## Model Evaluation Functions - Regression Module ##

##Imports

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from math import sqrt


#create the linear regression model and return a df with the x and y columns, baseline, 
#predictions, and residual calculations
def lr_model_with_residuals(df, x_col, y_col):
    df = df[[x_col, y_col]]
    df['yhat_baseline'] = df[y_col].mean()
    lr = LinearRegression(normalize=True)
    lr.fit(df[[x_col]], df[[y_col]])
    df['yhat'] = lr.predict(df[[x_col]])
    df['residual'] = df['yhat'] - df[y_col]
    df['residual_baseline'] = df['yhat_baseline'] - df[y_col]
    df['residual^2'] = df.residual**2
    df['residual_baseline^2'] = df.residual_baseline**2
    return df
    
# Plot residuals function
def plot_residuals(df, x_col, baseline_residual_col, model_residual_col):
    plt.figure(figsize=(12, 6))
    plt.subplot(121)
    plt.scatter(df[[x_col]], df[[baseline_residual_col]])
    plt.axhline(y = 0, ls = ':')
    plt.xlabel(x_col)
    plt.ylabel(baseline_residual_col)
    plt.title('Baseline Residuals')

    plt.subplot(122)
    plt.scatter(df[[x_col]], df[[model_residual_col]])
    plt.axhline(y = 0, ls = ':')
    plt.xlabel(x_col)
    plt.ylabel(model_residual_col)
    plt.title('OLS model residuals')
    return plt.show()

# Regression errors function
def regression_errors(df, residual2_col, residual_baseline2_col):
    SSE = float(df[residual2_col].sum())
    MSE = SSE/len(df)
    RMSE = sqrt(MSE)
    TSS = df[residual_baseline2_col].sum()
    ESS = TSS - SSE
    print(f'SSE: {SSE:.1f}')
    print(f'MSE: {MSE:.1f}')
    print(f'RMSE: {RMSE:.1f}')
    print(f'TSS: {TSS:.1f}')
    print(f'ESS: {ESS:.1f}')

# Baseline mean errors function
def baseline_mean_errors(df, residual_baseline2_col):
    SSE_baseline = df[residual_baseline2_col].sum()
    MSE_baseline = SSE_baseline/len(df)
    RMSE_baseline =  sqrt(MSE_baseline)
    print("SSE Baseline =", "{:.1f}".format(SSE_baseline))
    print("MSE baseline = ", "{:.1f}".format(MSE_baseline))
    print("RMSE baseline = ", "{:.1f}".format(RMSE_baseline))


# Better than baseline function
def better_than_baseline(df, residual2_col, residual_baseline2_col):
    print('=============================')
    print('Model Performance Evaluation:')
    print('-----------------------------')
    if (float(df[residual2_col].sum())) < (df[residual_baseline2_col].sum()):
        print('The model performs better than baseline')
    else:
        print('The model does not perform better than baseline')