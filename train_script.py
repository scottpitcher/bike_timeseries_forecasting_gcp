# train_script.py

import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

def main(args):
    # Load the preprocessed data
    df_cleaned = pd.read_csv(args.data_path)

    # Define features (X) and target (y)
    X = df_cleaned.drop(columns=['cnt'])
    y = df_cleaned['cnt']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train the model with hyperparameters
    model = GradientBoostingRegressor(
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        max_depth=args.max_depth,
        min_samples_split=args.min_samples_split,
        min_samples_leaf=args.min_samples_leaf,
        subsample=args.subsample
    )
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f'Model evaluation: MSE={mse}, R2={r2}')

    # Save the trained model
    joblib.dump(model, args.model_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', type=str, help='Path to the preprocessed data')
    parser.add_argument('--model-path', type=str, help='Path to save the trained model')
    parser.add_argument('--n_estimators', type=int, help='Number of estimators for the model')
    parser.add_argument('--learning_rate', type=float, help='Learning rate for the model')
    parser.add_argument('--max_depth', type=int, help='Max depth for the model')
    parser.add_argument('--min_samples_split', type=int, help='Min samples split for the model')
    parser.add_argument('--min_samples_leaf', type=int, help='Min samples leaf for the model')
    parser.add_argument('--subsample', type=float, help='Subsample for the model')
    args = parser.parse_args()
    main(args)
