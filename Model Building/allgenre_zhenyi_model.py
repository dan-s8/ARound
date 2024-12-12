# author: Zhenyi


# Import Libs:

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, make_scorer
from sklearn.model_selection import train_test_split, KFold, cross_val_score


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


# Note that Random Forest Reg cannot process MISSING values
# Here we have two ways to deal with the issue

# Solution 1: Try to fill missing values using an imputer

imputer = SimpleImputer(strategy='mean')  # Impute missing values
X_train = imputer.fit_transform(X_train)

# Solution 2: Simply drop them (the missing values)

X_train = X_train.dropna()  # drop NaN
y_train = y_train.loc[X_train.index]


# Define the Random Forest model
rf_model = RandomForestRegressor(
    n_estimators=100,  # Number of trees in the forest
    max_depth=None,    # Maximum depth of the tree (None means nodes are expanded until all leaves are pure)
    random_state=42,   # Random state for reproducibility
    n_jobs=-1          # Use all available
)

# Define scoring metrics
rmse_scorer = make_scorer(mean_squared_error, greater_is_better=False, squared=False)  # RMSE
r2_scorer = make_scorer(r2_score)  # R²


# CROSS VALIDATION
# Set up cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Perform cross-validation for RMSE
cv_rmse_scores = cross_val_score(rf_model, X_train, y_train, cv=kf, scoring=rmse_scorer)
formatted_rmse_scores = [int(round(-score)) for score in cv_rmse_scores]  # Negate each score, round, and convert to integer
mean_cv_rmse = int(round(-np.mean(cv_rmse_scores)))  # Negate the mean, round, and convert to integer

print("Cross-validation RMSE scores:", formatted_rmse_scores)
print("Mean CV RMSE:", mean_cv_rmse)

# Perform cross-validation for R²
cv_r2_scores = cross_val_score(rf_model, X_train, y_train, cv=kf, scoring=r2_scorer)
formatted_r2_scores = [round(score, 3) for score in cv_r2_scores]

print("Cross-validation R² scores:", formatted_r2_scores)
print("Mean CV R²:", round(np.mean(cv_r2_scores), 3))


# TEST ON TESTING SET
# Train the Random Forest model
rf_model.fit(X_train, y_train)

# Make predictions on the testing set
X_test = test_data[feature_columns]
y_test = test_data['Avg. Gross USD']
y_pred = rf_model.predict(X_test)

# Calculate R²
test_r2 = r2_score(y_test, y_pred)
print(f"Test R²: {test_r2:.3f}")

# Calculate RMSE
test_rmse = mean_squared_error(y_test, y_pred, squared=False)
print(f"Test RMSE: {test_rmse:.2f}")


# Solution 1:

# Cross validation results:
# Cross-validation RMSE scores: [197520, 160018, 170911, 160596, 195775]
# Mean CV RMSE: 176964
# Cross-validation R² scores: [0.92, 0.945, 0.945, 0.947, 0.941]
# Mean CV R²: 0.94

# Testing results:
# Test R²: 0.943
# Test RMSE: 157201.34


# Solution 2:

# Cross validation results:
# Cross-validation RMSE scores: [153027, 131186, 147266, 168443, 184163]
# Mean CV RMSE: 156817
# Cross-validation R² scores: [0.941, 0.949, 0.944, 0.92, 0.899]
# Mean CV R²: 0.931

# Testing results:
# Test R²: 0.934
# Test RMSE: 169654.83


# Random Forest has better metrics performance than linear reg
# Solution 1 (impute with mean values) has better metrics performance than Solution 2 (Drop NaN)
