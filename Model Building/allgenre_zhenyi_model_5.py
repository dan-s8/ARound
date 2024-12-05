# author: Zhenyi


# Import Libs:

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import make_scorer, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer


pd.set_option('display.max_columns', 60)  # make sure all cols can be displayed at the same time


# Load Data:

df = pd.read_csv('combined_df.csv')

# Note that 'Median age' is currently of Object type, we need to convert the string to a float
# df['Median age'] = df['Median age'].astype(float)  # 'Median age' should be of float64

# Filter the dataset without dropping NaN values (for training set)
filtered_data = df[
    (df['Year'] >= 2020) &
    (~(df['Headliner'].str.contains('"', na=False))) &
    (df['Genre'] != 'Family Entertainment') &
    (df['Ticket Price Min USD'] > 0) &
    (df['Ticket Price Min USD'] < df['Ticket Price Max USD'])
]

# Filter the dataset with NaN rows dropped (for testing set)
filtered_data_no_na = filtered_data.dropna()

# Split 30% of the cleaned dataset (no NaNs) into the testing set
_, test_data = train_test_split(filtered_data_no_na, test_size=0.3, random_state=42)

# Ensure the training set is mutually exclusive by removing test rows from the original filtered dataset
train_data = filtered_data.loc[~filtered_data.index.isin(test_data.index)]

# Check the results
print(f"Total filtered data size: {len(filtered_data)}")  # Total filtered data size: 79691
print(f"Training set size (including NaNs): {len(train_data)}")  # Training set size (including NaNs): 78682
print(f"Testing set size (no NaNs): {len(test_data)}")  # Testing set size (no NaNs): 1009


# APPLY ML MODEL:
# Extract features and target for the model
feature_columns = ['sp followers', 'sp popularity', 'yt View Count', 'yt Subscriber Count', 'Total population',
                   'monthly_listeners', 'Number of Shows', 'Avg. Event Capacity', 'Ticket Price Min USD', 'Ticket Price Max USD']
X_train = train_data[feature_columns]
y_train = train_data['Avg. Gross USD']

# Note that linear reg cannot process MISSING values
# Here we have two ways to deal with the issue

# Solution 1: Try to fill missing values using an imputer
# Impute missing values
imputer = SimpleImputer(strategy='mean')
X_train = imputer.fit_transform(X_train)

# Solution 2: Simply drop them (the missing values)
X_train = X_train.dropna()
y_train = y_train.loc[X_train.index]

# Define the Linear Regression model
lr_model = LinearRegression()

# Define scoring metrics
rmse_scorer = make_scorer(mean_squared_error, greater_is_better=False, squared=False)  # RMSE
r2_scorer = make_scorer(r2_score)  # R²


# CROSS VALIDATION
# Set up cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
# Perform cross-validation for RMSE
cv_rmse_scores = cross_val_score(lr_model, X_train, y_train, cv=kf, scoring=rmse_scorer)
formatted_rmse_scores = [int(round(-score)) for score in cv_rmse_scores]  # Negate each score, round, and convert to integer
mean_cv_rmse = int(round(-np.mean(cv_rmse_scores)))  # Negate the mean, round, and convert to integer
print("Cross-validation RMSE scores:", formatted_rmse_scores)
print("Mean CV RMSE:", mean_cv_rmse)
# Perform cross-validation for R²
cv_r2_scores = cross_val_score(lr_model, X_train, y_train, cv=kf, scoring=r2_scorer)
formatted_r2_scores = [round(score, 3) for score in cv_r2_scores]
print("Cross-validation R² scores:", formatted_r2_scores)
print("Mean CV R²:", round(np.mean(cv_r2_scores), 3))

# TEST ON TESTING SET
# Make predictions on the testing data
X_test = test_data[feature_columns]
y_test = test_data['Avg. Gross USD']
lr_model.fit(X_train, y_train)
y_pdt = lr_model.predict(X_test)
# Calculate R²
test_r2 = r2_score(y_test, y_pdt)
print(f"Test R²: {test_r2:.3f}")
# Calculate RMSE
test_rmse = mean_squared_error(y_test, y_pdt, squared=False)
print(f"Test RMSE: {test_rmse:.2f}")


# Solution 1:

# Cross validation results:
# Cross-validation RMSE scores: [70787201244, 349576, 393685, 370098, 413049]
# Mean CV RMSE: 14157745530
# Cross-validation R² scores: [-10229053090.326, 0.737, 0.71, 0.717, 0.736]
# Mean CV R²: -2045810617.485

# Testing results:
# Test R²: 0.620
# Test RMSE: 406626.63


# Solution 2:

# Cross validation results:
# Cross-validation RMSE scores: [308856, 299262, 301305, 496971, 294535]
# Mean CV RMSE: 340186
# Cross-validation R² scores: [0.759, 0.736, 0.766, 0.302, 0.743]
# Mean CV R²: 0.661

# Testing results:
# Test R²: 0.719
# Test RMSE: 350036.43


# Solution 2 (Drop NaN) has better metrics performance than Solution 1 (impute with mean values)
