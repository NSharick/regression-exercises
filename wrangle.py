#Wrangle Function - Regression Module

#Imports for the function:
import numpy as np
import pandas as pd
from env import get_db_url
import os

#clean zillow data function
def clean_zillow(df):
    df = df.dropna(subset =['bedroomcnt', 'calculatedfinishedsquarefeet', 'yearbuilt'])
    df['taxamount'].fillna(df.taxamount.median(), inplace=True)
    df['taxvaluedollarcnt'].fillna(df.taxvaluedollarcnt.median(), inplace=True)
    return df


#Wrangle Function for pulling zillow data and cleaning with above clean function
def wrangle_zillow():
    '''
    '''
    #assign the file name
    filename = 'zillow.csv'
    #check if the file exists in the current directory and read it if it is
    if os.path.exists(filename):
        print('Reading from csv file...')
        df = pd.read_csv(filename)
        return clean_zillow(df)
    #assign the sql query to a variable for use
    query = '''
    SELECT bedroomcnt, bathroomcnt, calculatedfinishedsquarefeet, taxvaluedollarcnt, yearbuilt, taxamount, fips
    FROM properties_2017
    WHERE propertylandusetypeid = 261
    '''
    #if needed pull a fresh copy of the dataset from the database and save localy
    print('Getting a fresh copy from SQL database...')
    df = pd.read_sql(query, get_db_url('zillow'))
    df.to_csv(filename, index=False)
    return clean_zillow(df)
    
    

