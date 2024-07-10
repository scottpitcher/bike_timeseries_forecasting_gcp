# pipeline.py

from kfp.v2 import dsl
from kfp.v2.dsl import component, Input, Output, Dataset, Model

@component
def data_prep_component():
    import subprocess
    subprocess.run(['python3', '1-data-prep-eda-component.py'], check=True)

@component
def model_initialization_component():
    import subprocess
    subprocess.run(['python3', '2-model-initialisation.py'], check=True)

@component
def hyperparameter_tuning_component():
    import subprocess
    subprocess.run(['python3', '3-tuning-component.py'], check=True)

@dsl.pipeline(
    name='bike-sharing-pipeline',
    description='An example pipeline that performs data preparation, model initialization, and hyperparameter tuning'
)
def bike_sharing_pipeline():
    # Data Preparation
    data_prep_task = data_prep_component()
    
    # Model Initialization
    model_initialization_task = model_initialization_component().after(data_prep_task)
    
    # Hyperparameter Tuning
    hyperparameter_tuning_task = hyperparameter_tuning_component().after(model_initialization_task)

if __name__ == '__main__':
    import kfp
    from kfp.v2 import compiler

    # Compile the pipeline
    compiler.Compiler().compile(
        pipeline_func=bike_sharing_pipeline,
        package_path='bike_sharing_pipeline.json'
    )

    # Set up the AI Platform client
    aiplatform.init(project='ultra-path-428923-i6', location='us-central1')

    # Create and run the pipeline job
    job = aiplatform.PipelineJob(
        display_name='bike_sharing_pipeline',
        template_path='bike_sharing_pipeline.json',
        pipeline_root='gs://your_bucket_name/pipeline_root'
    )

    job.run()
