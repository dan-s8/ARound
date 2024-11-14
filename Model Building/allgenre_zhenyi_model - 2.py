# Author: Zhenyi W
# ! Caution 1:
# ! Please only run one model a time! There are many models and variables in the code that may share names
# ! Comment out other models before you run a specific one, so that they won't mixing up something!
# ! Caution 2:
# ! The last part of the code (one hot coding) has unsolved bugs, run it will raise an error


# Import libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.tools.eval_measures import rmse
from sklearn import linear_model
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
# from sklearn.feature_selection import RFE  # comment out cuz it failed to improve the reg model


# Load the CSV file

# load as a data frame
file_path = 'Pollstar_all_genres.csv'
data = pd.read_csv(file_path)
pd.set_option('display.max_columns', 20)  # make sure all 20 cols can be displayed at the same time

# check null
print(data.isnull().sum())

# show first five rows
print(data.head())

# check data types
print(data.info())


# Data preprocessing

# Remove rows with missing values
df = data.iloc[:, -7:].dropna()

# make sure without null now
print(df.isnull().sum())
print(df.head())
print(df.info())

# Remove the '%' sign, convert the string to a float. For example, 100% will be changed to 100
df['Avg. Capacity Sold'] = df['Avg. Capacity Sold'].str.replace("%", "").astype(float)

# check it
print(df['Avg. Capacity Sold'].head())


# Explore features

# plot a corr heatmap for analyze
plt.figure()
sns.heatmap(df.corr(), annot=True, vmin=-1, vmax=1)
plt.show()
print(df.corr())


# Prepare for modeling

# create two dataframes for independent and dependent variables
X = df[['Avg. Tickets Sold', 'Avg. Event Capacity', 'Avg. Capacity Sold', 'Ticket Price Min USD', 'Ticket Price Max USD',
        'Ticket Price Avg. USD']]  # here we have 6 input variables for multiple regression.
Y = df['Avg. Gross USD']  # output variable (what we are trying to predict)


# liner regression model

# compute with sklearn
reg6 = linear_model.LinearRegression()
reg6.fit(X, Y)

# output Intercept and Coefficients
print('Intercept: \n', reg6.intercept_)  # -44736.46278683987
print('Coefficients: \n', reg6.coef_)  # [8.10677407e+01 -4.18344095e-03 -4.09760707e+02  1.60162520e+01 5.63454369e-07  5.84759370e+02]

# compute with statsmodels, by adding intercept manually
X1 = sm.add_constant(X)
result = sm.OLS(Y, X1).fit()

# OLS Regression Results
print(result.summary())

# calculate RMSE
yp = result.predict(X1)
ols_rmse = rmse(Y, yp)

# output RMSE and R^2
print(f'Root Mean Squared Error (RMSE): {ols_rmse}')  # Root Mean Squared Error (RMSE): 274295.8770138621
print(f'R-squared (R2) Score: {result.rsquared}')  # R-squared (R2) Score: 0.6026865951491364
# print(result.rsquared, result.rsquared_adj)  # 0.6026865951491364 0.6026832123045534


# # Feature selection

# # comment down cuz it failed to improve the reg model
# # tried to improve the model by selecting features but found it is not helpful
#
# # Create a model
# model = linear_model.LinearRegression()
#
# # Create an RFE object
# rfe5 = RFE(estimator=model, n_features_to_select=5)  # Select 5 features
# rfe5 = rfe5.fit(X, Y)
#
# # Get the selected features
# selected_features5 = X.columns[rfe5.support_]
# # print(selected_features5)
# X_5 = X[selected_features5]
# # # Xr = X['Avg. Tickets Sold', 'Avg. Event Capacity', 'Avg. Capacity Sold', 'Ticket Price Min USD', 'Ticket Price Avg. USD']
# # r = linear_model.LinearRegression()
# # r.fit(Xr, Y)
# # print(r.summary())
# X5 = sm.add_constant(X_5)
# result5 = sm.OLS(Y, X5).fit()
# print(result5.rsquared, result5.rsquared_adj)  # 0.6026865941214619 0.6026837750883025
#
# # Create an RFE object
# rfe4 = RFE(estimator=linear_model.LinearRegression(), n_features_to_select=4)  # Select 4 features
# rfe4 = rfe4.fit(X, Y)
#
# # Get the selected features
# selected_features4 = X.columns[rfe4.support_]
# print(selected_features4)  # ['Avg. Tickets Sold', 'Avg. Capacity Sold', 'Ticket Price Min USD', 'Ticket Price Avg. USD']
# X_4 = X[selected_features4]
# X4 = sm.add_constant(X_4)
# result4 = sm.OLS(Y, X4).fit()
# print(result4.rsquared, result4.rsquared_adj)  # 0.6026842037224485 0.6026819484855528
#
# # Create an RFE object
# rfe3 = RFE(estimator=linear_model.LinearRegression(), n_features_to_select=3)  # Select 3 features
# rfe3 = rfe3.fit(X, Y)
#
# # Get the selected features
# selected_features3 = X.columns[rfe3.support_]
# print(selected_features3)  # ['Avg. Tickets Sold', 'Avg. Capacity Sold', 'Ticket Price Avg. USD']
# X_3 = X[selected_features3]
# X3 = sm.add_constant(X_3)
# result3 = sm.OLS(Y, X3).fit()
# print(result3.rsquared, result3.rsquared_adj)  # 0.6026220357549037 0.602620344064975
#
# # Create an RFE object
# rfe2 = RFE(estimator=linear_model.LinearRegression(), n_features_to_select=2)  # Select 2 features
# rfe2 = rfe2.fit(X, Y)
#
# # Get the selected features
# selected_features2 = X.columns[rfe2.support_]
# print(selected_features2)  # ['Avg. Capacity Sold', 'Ticket Price Avg. USD']
# X_2 = X[selected_features2]
# X2 = sm.add_constant(X_2)
# result2 = sm.OLS(Y, X2).fit()
# print(result2.rsquared, result2.rsquared_adj)  # 0.06758777035623253 0.06758512409283957


# Random forest model

# Split the dataset into a training set and a testing set
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Create the Random Forest model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model
rf_model.fit(X_train, y_train)

# Predict on the test set
y_pred = rf_model.predict(X_test)

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

# Calculate R2 score
r2 = r2_score(y_test, y_pred)

print(f'Root Mean Squared Error (RMSE): {rmse}')  # Root Mean Squared Error (RMSE): 16786.035965380437
print(f'R-squared (R2) Score: {r2}')  # R-squared (R2) Score: 0.9983604296047138


# ---------------------------------------------------------------------------------------------------------------------


# Realized that we can't use avg price sold and capacity sold, etc.
# Hence, modify input as only has variables that not directly related to real sale results
Xa = df[['Avg. Event Capacity', 'Ticket Price Min USD', 'Ticket Price Max USD']]  # here we have 3 input variables.

# compute with statsmodels, by adding intercept manually
Xa1 = sm.add_constant(Xa)
result = sm.OLS(Y, Xa1).fit()
# print(result.summary())  # OLS Regression Results
yap = result.predict(Xa1)
# calc rmse
ols_rmse = rmse(Y, yap)

# Output RMSE and R2
print(f'Root Mean Squared Error (RMSE): {ols_rmse}')  # Root Mean Squared Error (RMSE): 434928.3343304379
print(f'R-squared (R2) Score: {result.rsquared}')  # R-squared (R2) Score: 0.0010814877150193691

# Obviously, this model has bad performance on r-squared. We should consider add more variables to predict.


# Found that number of shows may contribute to revenue, add it to the input variables
Xb = df[['Number of Shows', 'Avg. Event Capacity', 'Ticket Price Min USD', 'Ticket Price Max USD']]  # here we have 4 input variables.

# compute with statsmodels, by adding intercept manually
Xb1 = sm.add_constant(Xb)
result = sm.OLS(Y, Xb1).fit()
# print(result.summary())  # OLS Regression Results
ybp = result.predict(Xb1)
# calc rmse
ols_rmse = rmse(Y, ybp)

# Output RMSE and R2
print(f'Root Mean Squared Error (RMSE): {ols_rmse}')  # Root Mean Squared Error (RMSE): 434926.95343709947
print(f'R-squared (R2) Score: {result.rsquared}')  # R-squared (R2) Score: 0.0010878308185640062

# Still very low R2, nearly no difference between input with and without number of shows

# Redo random forest model as well using these 4 input variables

# Split the dataset into a training set and a testing set
X_train, X_test, y_train, y_test = train_test_split(Xb, Y, test_size=0.2, random_state=42)  # 4 input variables

# Create the Random Forest model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model
rf_model.fit(X_train, y_train)

# Predict on the test set
y_pred = rf_model.predict(X_test)

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

# Calculate R2 score
r2 = r2_score(y_test, y_pred)

print(f'Root Mean Squared Error (RMSE): {rmse}')  # Root Mean Squared Error (RMSE): 149515.9837664345
print(f'R-squared (R2) Score: {r2}')  # R-squared (R2) Score: 0.8699204533242271

# Good performance on R2, keep it first! Let's see any other way to model.


# So, this!
# After analyze the data again, found 'City' may related to the sale revenue.
# City is of object type. Try one hot coding to handle this non-numeric feature

# One hot coding on city
one_hot_df = pd.get_dummies(df, prefix=['City'])
Xo = one_hot_df.drop('Avg. Gross USD', axis=1)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(Xo, one_hot_df['Avg. Gross USD'], test_size=0.2, random_state=42)

# compute with sklearn
oh_reg = linear_model.LinearRegression()
oh_reg.fit(X_train, y_train)

# Make predictions on the test data
y_pred = oh_reg.predict(X_test)

# Calculate RMSE
rmse = mean_squared_error(y_test, y_pred, squared=False)

# Calculate R-squared
r2 = r2_score(y_test, y_pred)

print("Root Mean Squared Error (RMSE):", rmse)
print("R-squared:", r2)

# !Error: numpy.core._exceptions._ArrayMemoryError: Unable to allocate 17.4 GiB for an array with shape (704711, 3321) and data type float64

# Not sure the actual reason of the error. Try one hot coding again using another method

# Separate features and target
X = df.drop('Avg. Gross USD', axis=1)
y = df['Avg. Gross USD']

# Select columns that are not 'City'
Xn = X.drop('City', axis=1)  # all numeric input variables are here
# One-hot encode the 'City' column
encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
city_encoded = encoder.fit_transform(X[['City']])
city_encoded_df = pd.DataFrame(city_encoded, columns=encoder.get_feature_names(['City']))

# Combine the numeric features and one-hot encoded City features
X_processed = pd.concat([Xn, city_encoded_df], axis=1)

# Train Test split
X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)

# Liner reg model on the train set
model = linear_model.LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

# Calculate R-squared
r2 = r2_score(y_test, y_pred)

print("RMSE:", rmse)
print("R-squared:", r2)

# Error: numpy.core._exceptions._ArrayMemoryError: Unable to allocate 17.4 GiB for an array with shape (704711, 3321) and data type float64
# Since the error still here, one hot coding do unable to efficiently deal with 3321 unique cities in our dataset
# Will research on the issue next week and try to solve it.
