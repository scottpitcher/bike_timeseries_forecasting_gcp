# train_script.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import argparse

def train_and_save_model(data_path, model_path):
    # Load the preprocessed data
    df_cleaned = pd.read_csv(data_path)

    # Define features (X) and target (y)
    X = df_cleaned.drop(columns=['cnt'])
    y = df_cleaned['cnt']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train the model
    model = GradientBoostingRegressor()
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f'Model evaluation: MSE={mse}, R2={r2}')

    # Save the model
    joblib.dump(model, model_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', required=True, help='Path to the cleaned data CSV')
    parser.add_argument('--model-path', required=True, help='Path to save the trained model')
    args = parser.parse_args()

    train_and_save_model(args.data_path, args.model_path)
