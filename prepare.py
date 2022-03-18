#Split and scale functions for scaling exercise in regression module

#imports
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, QuantileTransformer

#data split function
def split_data(df):
    train_val, test = train_test_split(df, train_size = 0.8, random_state=123)
    train, validate = train_test_split(train_val, train_size = 0.7, random_state=123)
    return train, validate, test


#data scaling function
def min_max_scale(train, validate, test):
    #create the scaler object
    scaler = MinMaxScaler()
    #fit the scaler object with the train dataset
    scaler.fit(train)
    #scale the train dataset
    train_scaled = scaler.transform(train)
    #cast the train dataset as a pandas df
    train_scaled = pd.DataFrame(train_scaled, columns=train.columns, index=train.index)
    #scale the validate dataset
    val_scaled = scaler.transform(validate)
    #cast the validate dataset as a pandas df
    val_scaled = pd.DataFrame(val_scaled, columns=validate.columns, index=validate.index)
    #scale the test dataset
    test_scaled = scaler.transform(test)
    #cast the test dataset as a pandas df
    test_scaled = pd.DataFrame(test_scaled, columns=test.columns, index=test.index)
    #output the scaled datasets
    return train_scaled, val_scaled, test_scaled


