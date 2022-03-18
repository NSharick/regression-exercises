#Wrangle Function - Regression Module

#Imports for the function:
import numpy as np
import pandas as pd
from env import get_db_url
import os

#remove outliers function
def remove_outliers(df, k, col_list):
    ''' this function will remove outliers from a list of columns in a dataframe 
        and return that dataframe. A list of columns with significant outliers is 
        assigned to a variable in the below wrangle function and can be modified if needed
    '''
    #loop throught the columns in the list
    for col in col_list:
        q1, q3 = df[col].quantile([.25, .75])  # get quartiles
        iqr = q3 - q1   # calculate interquartile range
        upper_bound = q3 + k * iqr   # get upper bound
        lower_bound = q1 - k * iqr   # get lower bound
        # return dataframe without outliers
        df = df[(df[col] > lower_bound) & (df[col] < upper_bound)] 
    return df


#Wrangle Function for pulling zillow data and cleaning with above clean function
def wrangle_zillow():
    '''
    This function checks for a copy of the dataset in the local directory 
    and pulls a new copy and saves it if there is not one,
    it then cleans the data by removing significant outliers then
    removing the rows with null values for 'yearbuilt'
    '''
    #assign the file name
    filename = 'zillow.csv'
    #assign the columns to have outliers removed with the above function
    out_columns = ['bedroomcnt', 'bathroomcnt', 'calculatedfinishedsquarefeet',
       'taxvaluedollarcnt', 'taxamount']
    #check if the file exists in the current directory and read it if it is
    if os.path.exists(filename):
        print('Reading from csv file...')
        #read the local .csv into the notebook
        df = pd.read_csv(filename)
        #remove outliers from the identified columns 
        df = remove_outliers(df, 1.5, out_columns) 
        #drop remaining rows with null values for 'yearbuilt'
        df = df.dropna(subset=['yearbuilt'])
        #output the dataframe
        return df
    #assign the sql query to a variable for use in pulling a new copy of the dataset from the database
    query = '''
    SELECT bedroomcnt, bathroomcnt, calculatedfinishedsquarefeet, taxvaluedollarcnt, yearbuilt, taxamount, fips
    FROM properties_2017
    WHERE propertylandusetypeid = 261
    '''
    #if needed pull a fresh copy of the dataset from the database
    print('Getting a fresh copy from SQL database...')
    df = pd.read_sql(query, get_db_url('zillow'))
    #save a copy of the dataset to the local directory as a .csv file
    df.to_csv(filename, index=False)
    #remove outliers from the identified columns
    df = remove_outliers(df, 1.5, out_columns)  
    #drop the remaining rows with null values for 'yearbuilt'
    df = dropna(subset=['yearbuilt'])
    #output the dataframe
    return df
    

