from google.cloud import aiplatform
from google.cloud import storage
import os

class VertexAIManager:
    def __init__(self, project_id: str, region: str = "us-central1"):
        self.project_id = project_id
        self.region = region
        aiplatform.init(project=project_id, location=region)
        self.storage_client = storage.Client()
    
    def upload_model(self, local_path: str, bucket_name: str, blob_name: str):
        bucket = self.storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        blob.upload_from_filename(local_path)
        return f"gs://{bucket_name}/{blob_name}"
    
    def create_experiment(self, experiment_name: str):
        try:
            experiment = aiplatform.Experiment.create(
                experiment_name=experiment_name,
                description="Legal model fine-tuning experiment"
            )
            return experiment
        except Exception as e:
            print(f"Experiment might already exist: {e}")
    
    def log_metrics(self, metrics: dict):
        run = aiplatform.start_run("legal-model-run")
        for key, value in metrics.items():
            run.log_metric(key, value)
        run.end_run()
