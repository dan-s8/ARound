# author: Zhenyi


# Import Libs:
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, r2_score, make_scorer
from sklearn.model_selection import train_test_split, KFold, cross_val_score, RandomizedSearchCV

pd.set_option('display.max_columns', 60)  # make sure all cols can be displayed at the same time


# Load Data:
df = pd.read_csv('combined_df.csv')
# print(f"Total unfiltered data size: {len(df), len(df.columns)}")

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
# This week I'm trying XGBoost and LightGBM (for larger datasets and potentially better performance)
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

# Define scoring metrics
rmse_scorer = make_scorer(mean_squared_error, greater_is_better=False, squared=False)  # RMSE
r2_scorer = make_scorer(r2_score)  # R²

# Set up cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)


# LightGBM Model

# Define the LightGBM model
lgb_model = lgb.LGBMRegressor(objective='regression', random_state=42)

# Hyperparameter space for LightGBM
param_dist_lgb = {
    'n_estimators': [100, 200, 300, 400, 500],      # Increase max number of trees
    'max_depth': [None, 3, 5, 10, 12, 15, 20],       # Experiment with deeper trees
    'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3],   # Try smaller learning rates
    'subsample': [0.7, 0.8, 0.9, 1.0],               # Test smaller subsample
    'colsample_bytree': [0.7, 0.8, 0.9, 1.0],        # Test different feature fractions
    'min_child_samples': [5, 10, 20, 30, 50],         # Vary the leaf size constraint
    'num_leaves': [31, 50, 100, 150, 200]            # Test more leaves for flexibility
}

# RandomizedSearchCV for LightGBM
random_search_lgb = RandomizedSearchCV(lgb_model, param_distributions=param_dist_lgb,
                                       n_iter=10, cv=3, n_jobs=-1, random_state=42, verbose=2)
random_search_lgb.fit(X_train, y_train)

# Best LightGBM hyperparameters
best_params_lgb = random_search_lgb.best_params_
print(f"Best hyperparameters found by LightGBM RandomizedSearchCV: {best_params_lgb}")

# Get the best LightGBM model
best_lgb_model = random_search_lgb.best_estimator_

# Make predictions on the testing set
X_test = test_data[feature_columns]
y_test = test_data['Avg. Gross USD']

# Predict and evaluate on the test set
y_pred_lgb = best_lgb_model.predict(X_test)
lgb_r2 = r2_score(y_test, y_pred_lgb)
lgb_rmse = mean_squared_error(y_test, y_pred_lgb, squared=False)

print(f"LightGBM Model Test R²: {lgb_r2:.3f}")
print(f"LightGBM Model Test RMSE: {lgb_rmse:.2f}")

# Cross-validation on the training set with the best model
# Perform cross-validation for RMSE
cv_rmse_scores = cross_val_score(best_lgb_model, X_train, y_train, cv=kf, scoring=rmse_scorer)
formatted_rmse_scores = [int(round(-score)) for score in cv_rmse_scores]  # Negate each score, round, and convert to integer
mean_cv_rmse = int(round(-np.mean(cv_rmse_scores)))  # Negate the mean, round, and convert to integer

print("Cross-validation RMSE scores with best parameters:", formatted_rmse_scores)
print("Mean CV RMSE with best parameters:", mean_cv_rmse)

# Perform cross-validation for R²
cv_r2_scores = cross_val_score(best_lgb_model, X_train, y_train, cv=kf, scoring=r2_scorer)
formatted_r2_scores = [round(score, 3) for score in cv_r2_scores]

print("Cross-validation R² scores:", formatted_r2_scores)
print("Mean CV R²:", round(np.mean(cv_r2_scores), 3))


# ---- LightGBM Model Results ----
# Best hyper-parameters found by LightGBM RandomizedSearchCV: {'subsample': 0.9, 'num_leaves': 150, 'n_estimators': 500,
#                              'min_child_samples': 10, 'max_depth': 12, 'learning_rate': 0.01, 'colsample_bytree': 0.9}

# Cross-validation RMSE scores with best parameters: [135678, 139418, 146947, 147956, 180101]; Mean CV RMSE: 150020
# Cross-validation R² scores with best parameters: [0.953, 0.943, 0.944, 0.938, 0.904]; Mean CV R²: 0.936

# Test RMSE: 151189.62
# Test R²: 0.948
