# author: Zhenyi


# Import Libs:

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tools.eval_measures import rmse
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_selection import RFE
from sklearn import linear_model


pd.set_option('display.max_columns', 60)  # make sure all cols can be displayed at the same time


# Load Data:

df = pd.read_csv('combined_df.csv')
# DtypeWarning: Columns (2,3,6,7,8,9,10,28) have mixed types. Specify dtype option on import or set low_memory=False.
# print(df.info())  # RangeIndex: 707628 entries, 0 to 707627 total 53 columns


# Filter Data:

data = df[
    (df['Year'] >= 2020) &  # The Year must be 2020 or later
    (~(df['Headliner'].str.contains('"', na=False))) &  # The Headliner not festival(column contain double quotes)
    (df['Genre'] != 'Family Entertainment') &  # The Genre must not be "Family Entertainment" (that's sports)
    (df['Ticket Price Min USD'] > 0) &  # Ticket price should be more than 0
    (df['Ticket Price Min USD'] < df['Ticket Price Max USD'])  # The min price should be smaller than the Max price
]
# print(data.info())  # Index: 79691 entries, 0 to 109689 (total 53 columns)

data = data.dropna()  # all cols have no missing values after this
data['Event Date'] = pd.to_datetime(data['Event Date'], errors='coerce')
# print(data.info())  # Index: 3361 entries, 0 to 109068 total 53 columns

# Reset index after dropping rows
data = data.reset_index(drop=True)
print(data.info())  # RangeIndex: 3361 entries, 0 to 3360, dtypes: datetime64[ns](1), float64(29), int64(3), object(20)

# Note that 'Median age' is currently of Object type, we need to convert the string to a float
data['Median age'] = data['Median age'].astype(float)  # 'Median age' should be of float64

# check data types
print(data.dtypes)

# show first five rows
print(data.head())


# Feature Selection:

# Extract all numeric cols
X = data[['sp followers', 'sp popularity', 'yt View Count', 'yt Subscriber Count', 'yt Video Count', 'Total population',
          'Under 5 years population', '5 to 9 years population', '10 to 14 years population', '15 to 19 years population',
          '20 to 24 years population', '25 to 34 years population', '35 to 44 years population', '45 to 54 years population',
          '55 to 59 years population', '60 to 64 years population', '65 to 74 years population', '75 to 84 years population',
          '85 years and over population', 'Median age', 'Year', 'monthly_listeners', 'Number of Shows', 'Avg. Event Capacity',
          'Ticket Price Min USD', 'Ticket Price Max USD', 'Month', 'day_of_week']]  # gonna select from total 28 features
Y = data['Avg. Gross USD']

# Create a linear model
model = linear_model.LinearRegression()

# Split the dataset into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
X_train = X
X_test = X
y_train = Y
y_test = Y

# Create an RFE object
rfe5 = RFE(estimator=model, n_features_to_select=5)  # select 5 features

# Fit RFE on the training data
rfe5.fit(X_train, y_train)

# Get the selected features
selected_features5 = X_train.columns[rfe5.support_]

# Train the model using only the selected features
model.fit(X_train[selected_features5], y_train)

# Make predictions on the test data
y_predict = model.predict(X_test[selected_features5])

# Calculate RMSE
rm5 = np.sqrt(mean_squared_error(y_test, y_predict))

# Calculate R-squared
r5 = r2_score(y_test, y_predict)

# Get the selected features
print("Selected features:", selected_features5)
# Index(['sp popularity', 'Year', 'Number of Shows', 'Ticket Price Min USD', 'day_of_week'], dtype='object')

# Evaluate the model
print(f'Root Mean Squared Error (RMSE): {rm5}')  # Root Mean Squared Error (RMSE): 538005.9836011073
print(f'R-squared (R2) Score: {r5}')  # R-squared (R2) Score: 0.27230692315147575


# Repeat the Feature Selection process with n_features_to_select set to 10, we'll get a better result:

# Selected features: Index(['sp popularity', 'yt Video Count', 'Median age', 'Year', 'Number of Shows',
# 'Avg. Event Capacity', 'Ticket Price Min USD', 'Ticket Price Max USD', 'Month', 'day_of_week'], dtype='object')
# Root Mean Squared Error (RMSE): 330653.4945422477
# R-squared (R2) Score: 0.725134479132364


# Again with n_features_to_select set to 20. We will have:

# Selected features have previous 10 features(when n=10) and 10 segment year population features added:
# ['sp popularity', 'yt Video Count', 'Under 5 years population', '5 to 9 years population', '10 to 14 years population',
# '15 to 19 years population', '20 to 24 years population', '45 to 54 years population', '55 to 59 years population',
# '65 to 74 years population', '75 to 84 years population', '85 years and over population', 'Median age', 'Year',
# 'Number of Shows', 'Avg. Event Capacity', 'Ticket Price Min USD', 'Ticket Price Max USD', 'Month', 'day_of_week']
# Root Mean Squared Error (RMSE): 322259.02331065456
# R-squared (R2) Score: 0.7389136273486336


# Doubt that if metric difference exist between sm.OLS and linear_model.LinearRegression() method
# Try OLS using selected features (with n=10)
rfe10 = RFE(estimator=model, n_features_to_select=10)  # select 10 features
rfe10.fit(X_train, y_train)
X_10 = X[X_train.columns[rfe10.support_]]
X10 = sm.add_constant(X_10)
res10 = sm.OLS(Y, X10).fit()

# calculate RMSE
yp10 = res10.predict(X10)
rm10 = rmse(Y, yp10)

# output RMSE and R^2
print(f'Root Mean Squared Error (RMSE): {rm10}')  # Root Mean Squared Error (RMSE): 330653.49454224756
print(f'R-squared (R2) Score: {res10.rsquared}')  # R-squared (R2) Score: 0.7251344791323642

# Compared with linear_model.LinearRegression(), sm.OLS got nearly the same RMSE and R-squared results


# Alternative Model:

# Besides RFE, features also can be selected according to exploratory data analysis (EDA code omitted)
X_ = data[['sp followers', 'sp popularity', 'yt View Count', 'yt Subscriber Count', 'Total population', 'monthly_listeners',
           'Number of Shows', 'Avg. Event Capacity', 'Ticket Price Min USD', 'Ticket Price Max USD']]  # 10 input variables
Y = data['Avg. Gross USD']  # and 1 output variable (what we are trying to predict)

# liner regression model
X1 = sm.add_constant(X_)
res = sm.OLS(Y, X1).fit()

# OLS Regression Results
print(res.summary())

# calculate RMSE
yp = res.predict(X1)
rm = rmse(Y, yp)

# output RMSE and R^2
print(f'Root Mean Squared Error (RMSE): {rm}')  # Root Mean Squared Error (RMSE): 319143.02808762796
print(f'R-squared (R2) Score: {res.rsquared}')  # R-squared (R2) Score: 0.7439382234758498

# This model has better RMSE and R2 performance than the previous models
