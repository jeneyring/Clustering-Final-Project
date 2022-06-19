# Predicting Property Value for Zestimate scores at Zillow
## About the Project:
### Project Goals
Zillow is known as one of the number one resources in not only property listings but also for Zestimate property values. These value scores help home owners determine what their home is worth and also worth listing as.

When it comes to our current Zestimate property value scores, there are some major errors that are still occuring-- meaning our predictions here at Zillow can be anywhere from 100,000--350,000 off of what a home is really worth!

To help improve our current property value predictions model, I looked into some unstructured ML methods of cluster models to see if by clustering any of the key drivers that correlated to logerror, I could then create a model that improved our current model more. 

### Background Data information:
The data acquired in this project was collected from the MySQL database, that was based on the 2018 Kaggle competition. You can find out more about the origin of this dataset here: <a href="https://www.kaggle.com/competitions/zillow-prize-1/overview" title="Wikipedia">Kaggle's Zestimate Challenge.</a></li><br>

### Data Dictionary:
Some of the columns and datasets may not fully makes sense to those first meeting this data. Below is a table that helps define the features and terminology used in this project.

|Target|Datatype|Definition|
|:-------|:--------|:----------|
| zillow | 52441 non-null: float64 | Zillow's 2017 predictions & transactions dataset |

|Feature|Datatype|Definition|
|:-------|:--------|:----------|
| bedroomcnt          |  52441 non-null: float64 | bedroom count of 2017 data |
| bathroomcnt         |  52441 non-null: float64  | bathroom count of 2017 data|
| calculatedfinishedsqft   | 52441 non-null: float64 | finished property sqft|
| taxvaluedollarcnt       |  52441 non-null: float64 | Property Value |
| transactiondate   |  52441 non-null: object | date of property transactions|
| fips |  52441 non-null: float64 | non-null: float64| fips code (county/state)|

# Project Steps:
# __________________________________
## Acquire
Within the wrangle.py file there are functions that will help to:
- Pull in the 2017, Single Family Property residents with no duplicates
- Read the SQL acquire query to a csv file
- Assign the data to a variable
# ___________________________________
## Prepare
Within the wrangle.py file there are functions for the prepare stage that:
- Clean up the nulls by 60/60%
- Drop any unneccessary and duplicated columns
- Set the ranges for outliers based on CA housing/lot size requirements
- Splits data into train, validate and test
## Scale
The same wrangle.py file also holds scaling functions that:
- Use a MinMax scaler to scale train, validate and test data
- Drops logerror (target variable) from scaling functions
# ___________________________________
## Explore
Within the Final Report, there is a step-by-step breakdown of my Exploration methods, as well as the clusters created.
Cluster created and mentioned are:
- Cluster_features: this holds clusters on features of yearbuilt, sqft, and counties
- Cluster_LA: this holds cluster model on features only applicable to LA county
- Cluster_value: this holds a cluster model of features of taxvalue and calculated sqft.
- Cluster_prop_age_size: This holds the clusters of yearbuilt, calculsqft (minus county)
# ____________________________________
## Model
The Final_report notebook holds each method of model used in this project. Models include steps to produce: OLS| LassLars | GLM | Polynomial 2 & 3 degree.

The function for the final chosen model can be found in model.py
# ____________________________________
## Conclusion
Using the unstructured ML method of cluster models does not show to be the best model method when it comes to determining logerror predictions. 

If another cluster-model is wanted, I would suggest adding one more approach to the outliers handled, such as creating a logerror range to cutout the major outliers in logerror. 

I also think that the Regression model was the best approach and by adding in new features such as GreatSchools.com data, and a better 2016-2017 census source would improve the Zillow model by taking into account key features of the clients and their needs/income as well.

# ____________________________________
# To Reproduce:
- [x] Read this README.md
- [ ] Download wrangle.py , model.py, and Final_report.ipynb in your working directory.
- [ ] Run the Final_report.ipynb in your own Jupyter Notebook.
- [ ] Do your own exploring, modeling, etc.