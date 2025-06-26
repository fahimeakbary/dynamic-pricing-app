# Import necessary libraries
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_squared_error

# Load the dataset
data = pd.read_csv('data_modified.csv')

# Define features and target variable
# Exclude the target and any non-feature columns (like 'Churn' if not needed)
features = data.drop(columns=['MonthlyCharges', 'Churn'])
target = data['MonthlyCharges']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Set the MLflow experiment
mlflow.set_experiment("MonthlyCharges_Prediction")

# Start MLflow run
with mlflow.start_run():
    # Define the model and parameter grid
    model = RandomForestRegressor(random_state=42)
    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [None, 10, 20]
    }

    # Perform Grid Search
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_

    # Predict and evaluate
    y_pred = best_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** 0.5

    # Log parameters and metrics to MLflow
    mlflow.log_params(grid_search.best_params_)
    mlflow.log_metric("mse", mse)
    mlflow.log_metric("rmse", rmse)
    mlflow.sklearn.log_model(sk_model=best_model, name="random_forest_model")


    # Save the model using joblib
    model_path = 'best_model.pkl'
    joblib.dump(best_model, model_path)

    print(f"Model trained and logged with RMSE: {rmse:.2f}")
    print(f"Model saved as {model_path}")

