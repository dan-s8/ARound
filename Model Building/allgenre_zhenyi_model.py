# author: Zhenyi


# Import Libs:

import numpy as np
import pandas as pd
from scipy.stats import randint  # for RandomizedSearchCV use
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score, make_scorer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor  # two alternative Regs
from sklearn.model_selection import train_test_split, KFold, cross_val_score, RandomizedSearchCV  # instead of GridSearchCV


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


# Model 1: Revised Random Forest Model

# Define the Random Forest model
rf_model = RandomForestRegressor(
    random_state=42,  # Random state for reproducibility
    n_jobs=-1          # Use all available cores
)

# Define the hyper-parameter space (compared to GridSearchCV, smaller grid for RandomizedSearchCV)
param_dist = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': randint(2, 10),
    'min_samples_leaf': randint(1, 5),
    'max_features': ['sqrt', 'log2', None],
    'bootstrap': [True, False]
}

# Perform RandomizedSearchCV
random_search = RandomizedSearchCV(rf_model, param_distributions=param_dist,
                                   n_iter=10, cv=3, n_jobs=-1, random_state=42, verbose=2)

# Fit RandomSearchCV on the training data
random_search.fit(X_train, y_train)

# Get the best parameters
best_params_random = random_search.best_params_
print(f"Best hyper-parameters found by RandomizedSearchCV: {best_params_random}")

# Use the best model found by RandomizedSearchCV
best_rf_random_model = random_search.best_estimator_

# Make predictions on the testing set
X_test = test_data[feature_columns]
y_test = test_data['Avg. Gross USD']

# Predict and evaluate on the test set
y_pred_random = best_rf_random_model.predict(X_test)
random_r2 = r2_score(y_test, y_pred_random)
random_rmse = mean_squared_error(y_test, y_pred_random, squared=False)

print(f"RandomizedSearchCV Model Test R²: {random_r2:.3f}")
print(f"RandomizedSearchCV Model Test RMSE: {random_rmse:.2f}")


# Cross-validation on the training set with the best model
# Perform cross-validation for RMSE
cv_rmse_scores = cross_val_score(best_rf_random_model, X_train, y_train, cv=kf, scoring=rmse_scorer)
formatted_rmse_scores = [int(round(-score)) for score in cv_rmse_scores]  # Negate each score, round, and convert to integer
mean_cv_rmse = int(round(-np.mean(cv_rmse_scores)))  # Negate the mean, round, and convert to integer

print("Cross-validation RMSE scores with best parameters:", formatted_rmse_scores)
print("Mean CV RMSE with best parameters:", mean_cv_rmse)

# Perform cross-validation for R²
cv_r2_scores = cross_val_score(best_rf_random_model, X_train, y_train, cv=kf, scoring=r2_scorer)
formatted_r2_scores = [round(score, 3) for score in cv_r2_scores]

print("Cross-validation R² scores:", formatted_r2_scores)
print("Mean CV R²:", round(np.mean(cv_r2_scores), 3))


# Gradient Boosting can perform better than Random Forest in some cases, especially when fine-tuned.
# Model 2: Gradient Boosting Model

# Define the Gradient Boosting model
gb_model = GradientBoostingRegressor(random_state=42)

# Fit the model on the training data
gb_model.fit(X_train, y_train)

# Make predictions on the testing set
X_test = test_data[feature_columns]
y_test = test_data['Avg. Gross USD']

# Predict and evaluate on the test set
y_pred_gb = gb_model.predict(X_test)
gb_r2 = r2_score(y_test, y_pred_gb)
gb_rmse = mean_squared_error(y_test, y_pred_gb, squared=False)

print(f"Gradient Boosting Model Test R²: {gb_r2:.3f}")
print(f"Gradient Boosting Model Test RMSE: {gb_rmse:.2f}")


# Cross-validation on the training set with the best model
# Perform cross-validation for RMSE
cv_rmse_scores = cross_val_score(gb_model, X_train, y_train, cv=kf, scoring=rmse_scorer)
formatted_rmse_scores = [int(round(-score)) for score in cv_rmse_scores]  # Negate each score, round, and convert to integer
mean_cv_rmse = int(round(-np.mean(cv_rmse_scores)))  # Negate the mean, round, and convert to integer

print("Cross-validation RMSE scores with best parameters:", formatted_rmse_scores)
print("Mean CV RMSE with best parameters:", mean_cv_rmse)

# Perform cross-validation for R²
cv_r2_scores = cross_val_score(gb_model, X_train, y_train, cv=kf, scoring=r2_scorer)
formatted_r2_scores = [round(score, 3) for score in cv_r2_scores]

print("Cross-validation R² scores:", formatted_r2_scores)
print("Mean CV R²:", round(np.mean(cv_r2_scores), 3))


# Results:

# Model 1: Tried RandomizedSearchCV (instead of GridSearchCV used in previous code) and got similar R2 and RMSE

# Best hyper-parameters found by RandomizedSearchCV: {'bootstrap': True, 'max_depth': None, 'max_features': None, 'min_samples_leaf': 3, 'min_samples_split': 6, 'n_estimators': 100}
# RandomizedSearchCV Model Test R²: 0.933
# RandomizedSearchCV Model Test RMSE: 170920.05
# Cross-validation RMSE scores with best parameters: [159125, 135074, 143826, 171168, 189434]
# Mean CV RMSE with best parameters: 159725
# Cross-validation R² scores: [0.936, 0.946, 0.947, 0.917, 0.894]
# Mean CV R²: 0.928


# Model 2: Applied Gradient Boosting Model to predict but failed to improve R2 and RMSE significantly

# Gradient Boosting Model Test R²: 0.938
# Gradient Boosting Model Test RMSE: 164660.93
# Cross-validation RMSE scores with best parameters: [161727, 150037, 157804, 152175, 187264]
# Mean CV RMSE with best parameters: 161801
# Cross-validation R² scores: [0.934, 0.934, 0.936, 0.935, 0.896]
# Mean CV R²: 0.927
