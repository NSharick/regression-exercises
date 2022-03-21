## Exploration Exercise Functions for the Regression Module ##

#imports
import numpy as np
import pandas as pd
from env import get_db_url
import os
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt

#Get the Telco data from the database and join all four tables into the one dataframe
def get_telco_data():
    '''
    this function will check if there is already a copy of the dataset
    in the current working directory, read from that file if there is, 
    and pull a new copy from the database if there is not, it will also 
    save a copy to the local directory if it pulls a new one
    '''
    #assign the file name
    filename = 'telco.csv'
    #check if the file exists in the current directory and read it if it is
    if os.path.exists(filename):
        print('Reading from csv file...')
        return pd.read_csv(filename)
    #assign the sql query to a variable for use
    query = '''
    SELECT * FROM customers
    JOIN contract_types USING (contract_type_id)
    JOIN internet_service_types USING (internet_service_type_id)
    JOIN payment_types USING (payment_type_id)
    '''
    #if needed pull a fresh copy of the dataset from the database and save localy
    print('Getting a fresh copy from SQL database...')
    df = pd.read_sql(query, get_db_url('telco_churn'))
    df.to_csv(filename, index=False)
    return df  

#split the telco data function
def split_data(df):
    '''
    This function is for splitting the dataset into train, validate, and test 
    and is used in the preparation function below so that all data prep operations 
    can be done in the notebook with one call
    '''
    train_val, test = train_test_split(df, train_size = 0.8, stratify = df.churn, random_state=123)
    train, validate = train_test_split(train_val, train_size = 0.7, stratify = train_val.churn, random_state=123)
    return train, validate, test

#Prepare Telco dataset for modeling function
def prep_telco(df):
    '''
    This function prepares the dataset to be used for exploration 
    and for building machine learning models. This function keeps the original columns 
    after encoding the categorical variables for ml so that the original columns can be 
    used for data exploration
    '''
    #replace whitespace cells with np.nan so that calculations can be made with the column
    df.total_charges = df.total_charges.replace(' ', np.nan).astype(float)
    #drop foreign key id columns that resulted from the sql query
    df = df.drop(columns = ['payment_type_id', 'internet_service_type_id', 'contract_type_id'])
    #change the valuse in the churn column from "yes/no" to "1/0" for calculating churn rate etc.
    df['churn'] = df['churn'].apply(lambda x: 1 if x == 'Yes' else 0)
    #encode the categorical columns
    encode_cols = [col for col in df.columns if df[col].dtype == 'O']
    encode_cols.remove('customer_id')
    for col in encode_cols:
        dummie_df = pd.get_dummies(df[col], prefix = df[col].name, drop_first = True)
        df = pd.concat([df, dummie_df], axis=1)
    #rename the payment type columns to remove the "()" for functionality in python script
    df = df.rename(columns={'payment_type_Credit card (automatic)':'pay_credit', 'payment_type_Electronic check': 'pay_elec', 'payment_type_Mailed check': 'pay_mail'})
    #split the data using the above splitting function
    train, validate, test = split_data(df)
    #Return the train, validate, and test dataframes
    return train, validate, test

#plot variable pairs function
def plot_variable_pairs(columns):
    sns.pairplot(columns, kind='reg', plot_kws={'line_kws':{'color': 'red'}}, corner=True)
    return plt.show()  

#tenure months to full tenure years function
def tenure_full_years(df):
    df['tenure_years'] = (df.tenure / 12).astype(int)
    return df

#plot a categorical variable againt a continuous variable
def plot_categorical_and_continuous_vars(df, cat_col, cont_col):
    fig, ax = plt.subplots(ncols=3, figsize=(14, 8))
    sns.boxplot(x=cat_col, y=cont_col, data=df, ax=ax[0])
    sns.barplot(x=cat_col, y=cont_col, data=df, ax=ax[1])
    sns.violinplot(x=cat_col, y=cont_col, data=df, ax=ax[2])
    return plt.show()