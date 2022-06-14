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
import env

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
def handle_missing_values(df, prop_required_column = .5, prop_required_row = .75):
    threshold = int(round(prop_required_column*len(df.index),0))
    df.dropna(axis=1, thresh=threshold, inplace=True)
    threshold = int(round(prop_required_row*len(df.columns),0))
    df.dropna(axis=0, thresh=threshold, inplace=True)
    return df
#####_______________________________________
#PUTTING ABOVE TWO FUNCTIONS TOGETHER:
def data_prep(df, cols_to_remove=[], prop_required_column=.5, prop_required_row=.75):
    df = remove_columns(df, cols_to_remove)
    df = handle_missing_values(df, prop_required_column, prop_required_row)
    return df


