# author: Zhenyi


# Import Libs:

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, make_scorer
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV


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

# # Check the results
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

# # Solution 1: Try to fill missing values using an imputer
#
# imputer = SimpleImputer(strategy='mean')  # Impute missing values
# X_train = imputer.fit_transform(X_train)

# Solution 2: Simply drop them (the missing values)

X_train = X_train.dropna()  # drop NaN
y_train = y_train.loc[X_train.index]


# Define the Random Forest model
rf_model = RandomForestRegressor(
    random_state=42,  # Random state for reproducibility
    n_jobs=-1          # Use all available cores
)

# Define the hyper-parameter grid to search over
# Try this grid first (with just four parameters and two values for each parameter)
param_grid = {
        'n_estimators': [100, 200],  # Reduce the number of trees to test
        'max_depth': [None, 10],      # Limit max depth to None or a small depth (10)
        'min_samples_split': [2, 5],  # Test only two values
        'min_samples_leaf': [1, 2],   # Test only two values
}

# # Caution: This grid leads to memory error, which is currently unsolved
# param_grid = {
#     'n_estimators': [100, 500, 1000],  # Number of trees
#     'max_depth': [None, 10, 20, 30],    # Max depth of each tree
#     'min_samples_split': [2, 5, 10],    # Minimum samples required to split a node
#     'min_samples_leaf': [1, 2, 4],      # Minimum samples required at each leaf node
#     'max_features': ['auto', 'sqrt', 'log2'],  # Number of features to consider for the best split
#     'bootstrap': [True, False]  # Whether to use bootstrap samples (True) or not (False)
# }

# CROSS VALIDATION
# Set up cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Set up GridSearchCV
grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid,
                           cv=kf, n_jobs=-1, verbose=2, scoring='neg_mean_squared_error', return_train_score=True)

# Fit GridSearchCV on the training data
grid_search.fit(X_train, y_train)

# Get the best hyperparameters from the grid search
best_params = grid_search.best_params_
print(f"Best hyperparameters found: {best_params}")  # {'max_depth': None, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 100}

# Train the Random Forest model with the best parameters
best_rf_model = grid_search.best_estimator_

# Make predictions on the testing set
X_test = test_data[feature_columns]
y_test = test_data['Avg. Gross USD']
y_pred = best_rf_model.predict(X_test)

# Calculate and print performance metrics (R² and RMSE)
test_r2 = r2_score(y_test, y_pred)
test_rmse = mean_squared_error(y_test, y_pred, squared=False)
print(f"Test R² with best hyperparameters: {test_r2:.3f}")
print(f"Test RMSE with best hyperparameters: {test_rmse:.2f}")

# Define scoring metrics
rmse_scorer = make_scorer(mean_squared_error, greater_is_better=False, squared=False)  # RMSE
r2_scorer = make_scorer(r2_score)  # R²

# Cross-validation on the training set with the best model
# Perform cross-validation for RMSE
cv_rmse_scores = cross_val_score(best_rf_model, X_train, y_train, cv=kf, scoring=rmse_scorer)
formatted_rmse_scores = [int(round(-score)) for score in cv_rmse_scores]  # Negate each score, round, and convert to integer
mean_cv_rmse = int(round(-np.mean(cv_rmse_scores)))  # Negate the mean, round, and convert to integer

print("Cross-validation RMSE scores with best parameters:", formatted_rmse_scores)
print("Mean CV RMSE with best parameters:", mean_cv_rmse)

# Perform cross-validation for R²
cv_r2_scores = cross_val_score(best_rf_model, X_train, y_train, cv=kf, scoring=r2_scorer)
formatted_r2_scores = [round(score, 3) for score in cv_r2_scores]

print("Cross-validation R² scores:", formatted_r2_scores)
print("Mean CV R²:", round(np.mean(cv_r2_scores), 3))


# Got the same results as previous RF model (without grid search). It indicates:
# The current model configuration is already optimal, further hyper-parameter tuning may not provide significant improvements.
# The problem may not be the hyper-parameters, but something else...
# Such as the features, the quality of the data, or the inherent nature of the model for the problem we're trying to solve.
