##Functions for the feature engineering exercises - regression module##

##Imports##
import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, RFE, f_regression, SequentialFeatureSelector
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, QuantileTransformer

#K Best - feature selection function
def select_k_best(x, y, k):
    #create the kbest object
    kbest = SelectKBest(f_regression, k=k)
    #fit the kbest object with the input df from the function
    kbest.fit(x, y)
    #output the results
    kbest_results = pd.DataFrame(dict(p=kbest.pvalues_, f=kbest.scores_), index=x.columns)
    print(f'The {k} best features = {list(x.columns[kbest.get_support()])}')
    return kbest_results

#recursive feature elimination - feature selection function
def rfe(x, y, n_features):
    #create the model object
    model = LinearRegression()
    #create the rfe object
    rfe = RFE(model, n_features_to_select=n_features)
    #fit the rfe object
    rfe.fit(x, y)
    #output the results
    print(f'The {n_features} best features = {list(x.columns[rfe.get_support()])}')
    return pd.DataFrame({'rfe_ranking': rfe.ranking_}, index=x.columns)