import requests
import os

# api/utils/model_utils.py

from google.cloud import storage
import os

def download_from_gcs(bucket_name: str, gcs_path: str, local_path: str):
    if os.path.exists(local_path):
        print(f"[INFO] Model already exists at {local_path}")
        return

    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    print(f"[INFO] Downloading model from GCS: gs://{bucket_name}/{gcs_path}")

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(gcs_path)
    blob.download_to_filename(local_path)

    print(f"[INFO] Model downloaded and saved to {local_path}")
