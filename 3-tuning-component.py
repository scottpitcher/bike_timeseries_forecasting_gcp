# 3_model_training_hyperparameter_tuning.py

from google.cloud import aiplatform, storage
from google.cloud.aiplatform import hyperparameter_tuning as hpt

def hyperparameter_tuning():
    # Initialize Vertex AI
    aiplatform.init(project='ultra-path-428923-i6', location='us-central1')

    # Download the initial model from Google Cloud Storage to ensure it exists
    storage_client = storage.Client()
    bucket_name = 'timeseries_forecasting'
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob('model.pkl')
    blob.download_to_filename('model.pkl')

    # Define the hyperparameter tuning job
    job = aiplatform.HyperparameterTuningJob(
        display_name='bike_sharing_hyperparameter_tuning',
        
        custom_job=aiplatform.CustomJob(
            display_name='bike_sharing_training',
            script_path='train_script.py',  # Path to the training script
            container_uri='us-docker.pkg.dev/vertex-ai/training/scikit-learn-cpu.0-23:latest',
            requirements=['pandas', 'scikit-learn'],
            args=[
                '--data-path', 'cleaned_bike_data.csv', 
                '--model-path', 'initial_model.pkl'
            ]
        ),
        
        metric_spec={'mean_squared_error': 'MINIMIZE'},
        
        parameter_spec={
            'n_estimators': hpt.IntegerParameterSpec(min=300, max=400, scale='UNIT_LINEAR_SCALE'),
            'learning_rate': hpt.DoubleParameterSpec(min=0.1, max=0.2, scale='UNIT_LINEAR_SCALE'),
            'max_depth': hpt.IntegerParameterSpec(min=4, max=4, scale='UNIT_LINEAR_SCALE'),
            'min_samples_split': hpt.IntegerParameterSpec(min=2, max=10, scale='UNIT_LINEAR_SCALE'),
            'min_samples_leaf': hpt.IntegerParameterSpec(min=4, max=6, scale='UNIT_LINEAR_SCALE'),
            'subsample': hpt.DoubleParameterSpec(min=0.9, max=1.0, scale='UNIT_LINEAR_SCALE')
        },
        max_trial_count=10,
        parallel_trial_count=2,
    )

    job.run()

    print("Hyperparameter tuning completed successfully.")
    
    # Find the best trial
    best_trial = job.trials[0]
    for trial in job.trials:
        if trial.final_measurement.metrics['mean_squared_error'] < best_trial.final_measurement.metrics['mean_squared_error']:
            best_trial = trial

    # The best model path in the best trial's output
    best_model_path = 'model_output/model.pkl'

    # Initialize storage client
    storage_client = storage.Client()

    # Define bucket name and blob name
    bucket_name = 'timeseries_forecasting'
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob('tuned_model.pkl')

    # Upload the best model to Google Cloud Storage
    blob.upload_from_filename(best_model_path)

    print("Best model uploaded to Google Cloud Storage successfully as tuned_model.pkl.")




if __name__ == '__main__':
    hyperparameter_tuning()
