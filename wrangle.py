"""
This file holds the functions that pair along the Zillow Clustering Final Project
by Jen Eyring, Jemison-cohort.

These functions are for:
-Acquiring the dataset from MySQL
-Preparing the dataset through handling nulls, dropping columns, etc
-Split the dataset into Train, Validate, Test

"""

#imports for functions to work:
import os
import pandas as pd
import numpy as np
import env
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

sql = """SELECT
        p.*,
        predictions_2017.logerror,
        predictions_2017.transactiondate,
        a.airconditioningdesc,
        arch.architecturalstyledesc,
        build.buildingclassdesc,
        heat.heatingorsystemdesc,
        landuse.propertylandusedesc,
        story.storydesc,
        construct.typeconstructiondesc
    FROM properties_2017 AS p
    JOIN (
        SELECT parcelid, MAX(transactiondate) AS max_transactiondate
        FROM predictions_2017
        GROUP BY parcelid
    ) pred USING(parcelid)
    JOIN predictions_2017 ON pred.parcelid = predictions_2017.parcelid
                          AND pred.max_transactiondate = predictions_2017.transactiondate
    LEFT JOIN airconditioningtype AS a USING (airconditioningtypeid)
    LEFT JOIN architecturalstyletype arch USING (architecturalstyletypeid)
    LEFT JOIN buildingclasstype build USING (buildingclasstypeid)
    LEFT JOIN heatingorsystemtype heat USING (heatingorsystemtypeid)
    LEFT JOIN propertylandusetype landuse USING (propertylandusetypeid)
    LEFT JOIN storytype story USING (storytypeid)
    LEFT JOIN typeconstructiontype construct USING (typeconstructiontypeid)
    WHERE p.latitude IS NOT NULL
      AND p.longitude IS NOT NULL
      AND transactiondate <= '2017-12-31'
      AND propertylandusedesc = "Single Family Residential"
"""
def get_zillow_data():
    if os.path.exists('zillow.csv'):
        df = pd.read_csv('zillow.csv')
    else:
        database = 'zillow'
        url = f'mysql+pymysql://{env.username}:{env.password}@{env.host}/{database}'
        df = pd.read_sql(sql, url)
        df.to_csv('zillow.csv', index=False)
    return df

def df_summary(df):
    print("---Shape: {}".format(df.shape))
    print()
    print('---Info')
    df.info()
    print()
    print('--- Column Description')
    print(df.describe(include='all'))
#####______________________________________
#how to look at count and percentage of nulls per column
def nulls_by_columns(df):
    return pd.concat([
        df.isna().sum().rename('count'),
        df.isna().mean().rename('percent')
    ], axis=1)

#how to look at count and percentage of nulls per row
def nulls_by_rows(df):
    return pd.concat([
        df.isna().sum(axis=1).rename('n_missing'),
        df.isna().mean(axis=1).rename('percent_missing'),
    ], axis=1).value_counts().sort_index()

#####______________________________________
#removing columns not needed/wanted/replicated
def remove_columns(df, cols_to_remove):
    df = df.drop(columns=cols_to_remove)
    return df

#How to handle missing values based on minimum percentage of values 
#for rows and columns
def handle_missing_values(df, prop_required_column = .6, prop_required_row = .6):
    threshold = int(round(prop_required_column*len(df.index),0))
    df.dropna(axis=1, thresh=threshold, inplace=True)
    threshold = int(round(prop_required_row*len(df.columns),0))
    df.dropna(axis=0, thresh=threshold, inplace=True)
    return df

#dropping the remaining nulls
def drop_r_nulls(df):
    df = df.dropna()
    return df



#####_______________________________________
#PUTTING ABOVE CLEAN/PREP FUNCTIONS TOGETHER:
def data_prep(df, cols_to_remove=[], prop_required_column=.6, prop_required_row=.6):
    df = remove_columns(df, cols_to_remove)
    df = handle_missing_values(df, prop_required_column, prop_required_row)
    df = drop_r_nulls(df)
    return df

#####_____________________________________
#Splitting the Data:
def split_data(df):
    '''
    take in a DataFrame and return train, validate, and test DataFrames; stratify on species.
    return train, validate, test DataFrames.
    '''
    
    # splits df into train_validate and test using train_test_split() stratifying on fips to get an even mix of each county
    train_validate, test = train_test_split(df, test_size=.2, random_state=123, stratify=df.fips)
    
    # splits train_validate into train and validate using train_test_split() stratifying on fips to get an even mix of each county
    train, validate = train_test_split(train_validate, 
                                       test_size=.3, 
                                       random_state=123, 
                                       stratify=train_validate.fips)
    return train, validate, test

def handle_outliers(df):
    """Manually handle outliers that do not represent properties likely for the larger majority of buyers and zillow visitors"""
    #max/min of calculatedsqft
    df = df[df.calculatedfinishedsquarefeet <= 9_000]
    df = df[df.calculatedfinishedsquarefeet >= 200]
    #max/min of bedroomcnt
    df = df[df.bedroomcnt <= 6]
    df = df[df.bedroomcnt != 0]
    #max/min of bathroomcnt
    df = df[df.bathroomcnt <= 6]
    df = df[df.bathroomcnt != 0]
    #max/min of taxvaluedollar
    df = df[df.taxvaluedollarcnt <= 2_500_000]
    df = df[df.taxvaluedollarcnt >= 45_000]
    #max/min of yearbuilt
    df = df[df.yearbuilt <= 2016]
    df = df[df.yearbuilt >= 1950]
    #max/min of lot size
    df = df[df.lotsizesquarefeet <= 435_600]
    df = df[df.lotsizesquarefeet >= 2_500]

    return df


#to use, type 'split_data('add your df here')

######____________________________________________
#Encoding fips
#encode fips
def one_hot_encode(train, validate, test):
    train['is_Los_Angeles'] = train.fips == 6037.0
    validate['is_Los_Angeles'] = validate.fips == 6037.0
    test['is_Los_Angeles'] = test.fips == 6037.0
    
    train['is_Ventura'] = train.fips == 6111.0
    validate['is_Ventura'] = validate.fips == 6111.0
    test['is_Ventura'] = test.fips == 6111.0
    
    train['is_Orange'] = train.fips == 6059.0
    validate['is_Orange'] = validate.fips == 6059.0
    test['is_Orange'] = test.fips == 6059.0
    
    return train, validate, test


def dtype_county(train, validate, test):
    train["is_Los_Angeles"] = train["is_Los_Angeles"].astype(int)
    validate["is_Los_Angeles"] = validate["is_Los_Angeles"].astype(int)
    test["is_Los_Angeles"] = test["is_Los_Angeles"].astype(int)

    train['is_Ventura'] = train['is_Ventura'].astype(int)
    validate['is_Ventura'] = validate['is_Ventura'].astype(int)
    test['is_Ventura'] = test['is_Ventura'].astype(int)


    train['is_Orange'] = train['is_Orange'].astype(int)
    validate['is_Orange'] = validate['is_Orange'].astype(int)
    test['is_Orange'] = test['is_Orange'].astype(int)
    return train, validate, test

### Adding above together: Use after having created split of train/validate/test

def data_formats(train, validate, test):
    train = one_hot_encode(train, validate, test)
    train = dtype_county(train, validate, test)
    return train, validate, test

### Functions for Scaling data and specific columns for Zillow dataset
def scale(train, validate, test):
    columns_to_scale = [ 'yearbuilt', 'latitude', 'longitude','lotsizesquarefeet' ,'taxvaluedollarcnt', 'calculatedfinishedsquarefeet']
    train_scaled = train.copy()
    validate_scaled = validate.copy()
    test_scaled = test.copy()

    scaler = MinMaxScaler()
    scaler.fit(train[columns_to_scale])

    train_scaled[columns_to_scale] = scaler.transform(train[columns_to_scale])
    validate_scaled[columns_to_scale] = scaler.transform(validate[columns_to_scale])
    test_scaled[columns_to_scale] = scaler.transform(test[columns_to_scale])

    return scaler, train_scaled, validate_scaled, test_scaled