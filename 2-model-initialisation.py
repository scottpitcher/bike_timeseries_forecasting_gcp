# 2_model_initialization.py

from google.cloud import bigquery, storage
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os

def initialize_model():
    # Set up BigQuery client
    client = bigquery.Client()

    # Define a query to load the cleaned data from BigQuery
    query = """
    SELECT *
    FROM `ultra-path-428923-i6.london_bikes.london_bikes`
    """

    # Execute the query and load data into a pandas DataFrame
    df = client.query(query).to_dataframe()

    # Drop non-numeric columns and columns not needed for model training
    df = df.drop(columns=['timestamp'])

    # Ensure all columns are numeric
    df = df.apply(pd.to_numeric, errors='coerce')

    # Define features (X) and target (y)
    X = df.drop(columns=['cnt'])
    y = df['cnt']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize models
    models = {
        'Linear Regression': LinearRegression(),
        'Decision Tree': DecisionTreeRegressor(),
        'Random Forest': RandomForestRegressor(),
        'Gradient Boosting': GradientBoostingRegressor()
    }

    # Evaluate models
    best_model = None
    best_score = float('-inf')

    for name, model in models.items():
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        print(f"{name}: R2 Score={score}")
        if score > best_score:
            best_score = score
            best_model = model

    # Evaluate the best model
    y_pred = best_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f'Best Model: {type(best_model).__name__}')
    print(f'Model evaluation: MSE={mse}, R2={r2}')
    
    # Create a folder for storing the models
    models_folder = 'models'
    os.makedirs(models_folder, exist_ok=True)
    
    # Save the best model
    joblib.dump(best_model, f'{models_folder}/initial_model.pkl')

    # Upload the initial model to Google Cloud Storage
    storage_client = storage.Client()
    bucket_name = 'timeseries_forecasting'
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob('initial_model.pkl')
    blob.upload_from_filename(f'{models_folder}/initial_model.pkl')

    print("Initial model uploaded to Google Cloud Storage successfully.")

if __name__ == '__main__':
    initialize_model()
