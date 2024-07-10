# 1_data_prep.py

from google.cloud import bigquery
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

def data_prep():
    # Set up BigQuery client
    client = bigquery.Client()

    # Define a query to load the cleaned data from BigQuery
    query = """
    SELECT *
    FROM `ultra-path-428923-i6.london_bikes.london_bikes`
    """

    # Execute the query and load data into a pandas DataFrame
    df = client.query(query).to_dataframe()

    # Save the dataframe as a CSV file for further processing
    os.makedirs('data', exist_ok=True)
    df.to_csv('data/cleaned_bike_data.csv', index=False)

    # High-level statistical analysis
    def perform_eda(data):
        print("Performing exploratory data analysis...")

        # Display the first few rows of the dataset
        print("First few rows of the dataset:")
        print(data.head())

        # Display summary statistics
        print("\nSummary statistics:")
        print(data.describe())

        # Display information about the dataset
        print("\nData information:")
        print(data.info())

        # Check for missing values
        print("\nMissing values:")
        print(data.isnull().sum())

        # Correlation matrix
        print("\nCorrelation matrix:")
        corr_matrix = data.corr()
        print(corr_matrix)
        
        # Create a folder for storing the visuals
        visuals_folder = 'visuals'
        os.makedirs(visuals_folder, exist_ok=True)

        # Visualize the correlation matrix
        plt.figure(figsize=(12, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
        plt.title("Correlation Matrix")
        plt.savefig(os.path.join(visuals_folder, 'correlation_heatmap.png'))
        plt.close()

        # # Visualize the distribution of the target variable 'cnt'
        # plt.figure(figsize=(10, 6))
        # sns.histplot(data['cnt'], bins=30, kde=True, element="step")
        # plt.title("Distribution of Bike Counts")
        # plt.xlabel("Bike Counts")
        # plt.ylabel("Frequency")
        # plt.savefig(os.path.join(visuals_folder, 'bike_count_dist.png'))
        # plt.close()

        # Visualize relationships between target and important features
        important_features = ['t1', 't2', 'hum', 'wind_speed', 'is_holiday', 'is_weekend', 'season']

        for feature in important_features:
            plt.figure(figsize=(10, 6))
            sns.scatterplot(x=data[feature], y=data['cnt'])
            plt.title(f"Bike Counts vs {feature.capitalize()}")
            plt.xlabel(feature.capitalize())
            plt.ylabel("Bike Counts")
            plt.savefig(os.path.join(visuals_folder, f'feature_{feature}.png'))
            plt.close()

        print("Exploratory data analysis completed.")

    # Perform exploratory data analysis
    perform_eda(df)

    print("Data preparation completed successfully.")

if __name__ == '__main__':
    data_prep()
