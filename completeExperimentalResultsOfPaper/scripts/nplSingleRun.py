import os
import sys
import json
import argparse
import numpy as np
import time

try:
    from google.cloud import storage
except ImportError:
    storage = None

from NPL import npl
from models import g_and_k_model

def download_from_gcs(bucket_name, source_blob_name, destination_file_name):
    """Downloads a blob from the bucket."""
    if storage is None:
        print("google-cloud-storage not installed. Cannot download from GCS.")
        return False
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(source_blob_name)
        blob.download_to_filename(destination_file_name)
        print(f"Downloaded storage object {source_blob_name} from bucket {bucket_name} to local file {destination_file_name}.")
        return True
    except Exception as e:
        print(f"Error downloading from GCS: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Run NPL MMD for a Single Dataset")
    
    # Task Identification
    parser.add_argument('--task_id', type=int, required=True, help='Dataset index (0-19)')
    
    # Ground Truth
    parser.add_argument('--a', type=float, default=3.0, help='True parameter a')
    parser.add_argument('--b', type=float, default=1.0, help='True parameter b')
    parser.add_argument('--g', type=float, default=2.0, help='True parameter g')
    parser.add_argument('--k', type=float, default=0.5, help='True parameter k')
    
    # Study Params
    parser.add_argument('--N', type=int, default=5000, help='Sample size N')
    parser.add_argument('--bandwidth', type=float, default=-1.0, help='RBF Kernel bandwidth')
    parser.add_argument('--B', type=int, default=100, help='Number of bootstrap samples')
    parser.add_argument('--m', type=int, default=500, help='Number of simulated samples')
    parser.add_argument('--batch_size', type=int, default=500, help='Mini-batch size')
    
    # Paths and Config
    parser.add_argument('--data_dir', type=str, required=True, help='GCS path (gs://...) or local path to datasets')
    parser.add_argument('--bucket', type=str, default='sbi-sims-data-results', help='GCS bucket for output')
    
    args = parser.parse_args()

    # --- Data Fetching ---
    dataset_index = args.task_id
    base_path = args.data_dir
    
    # Strip trailing slashes or formatting
    if base_path.endswith("/"):
        base_path = base_path[:-1]
        
    full_gcs_path = f"{base_path}/dataset_{dataset_index}.npy"
    local_data_path = f"data_temp_{args.task_id}.npy"
    
    if full_gcs_path.startswith("gs://"):
        # e.g., gs://sbi-sims-data-results/gandk/experimentalData/N5000/dataset_0.npy
        parts = full_gcs_path.split("/")
        bucket_candidate = parts[2]
        gcs_blob_path = "/".join(parts[3:])
        if not download_from_gcs(bucket_candidate, gcs_blob_path, local_data_path):
            print(f"Failed to download dataset from {full_gcs_path}")
            sys.exit(1)
    else:
        local_data_path = full_gcs_path
        if not os.path.exists(local_data_path):
            print(f"Local data file not found: {local_data_path}")
            sys.exit(1)
            
    # Load data
    print(f"Loading data from {local_data_path}")
    X = np.load(local_data_path)
    if len(X.shape) == 1:
        X = X.reshape(-1, 1)

    # --- Initialization ---
    model = g_and_k_model(m=args.m, d=1)
    
    npl_inference = npl(X=X, 
                        B=args.B, 
                        m=args.m, 
                        p=4, 
                        l=args.bandwidth, 
                        model=model, 
                        model_name='gandk',
                        batch_size=args.batch_size)

    # --- Run Bayesian Bootstrap ---
    print(f"Starting NPL execution for dataset {dataset_index} with B={args.B} bootstraps...")
    start_time = time.time()
    npl_inference.draw_samples()
    samples = np.array(npl_inference.sample) # Shape (B, 4)
    elapsed = time.time() - start_time
    print(f"NPL completed in {elapsed:.2f} seconds.")

    # --- Upload Results to GCS ---
    bw_str = str(int(args.bandwidth)) if args.bandwidth == int(args.bandwidth) else str(args.bandwidth)
    
    # Setup GCS client for writing results
    client = None
    target_bucket = None
    if storage is not None:
        try:
            client = storage.Client()
            target_bucket = client.bucket(args.bucket)
        except Exception as e:
            print(f"Warning: Failed to initialize GCS client: {e}")
            
    # Save B individual task JSONs
    for b in range(args.B):
        global_task_id = args.task_id * 100 + b
        
        result = {
            "task_id": global_task_id,
            "dataset_id": dataset_index,
            "theta_est": samples[b].tolist(),
            "final_f": 0.0,
            "params": vars(args)
        }
        
        # Local save
        output_filename = f"task_{global_task_id}.json"
        with open(output_filename, 'w') as f:
            json.dump(result, f)
            
        # GCS upload
        if target_bucket is not None:
            gcs_results_path = f"gandk/results/npl_mmd/N{args.N}/Bandwidth={bw_str}/task_{global_task_id}.json"
            try:
                blob = target_bucket.blob(gcs_results_path)
                blob.upload_from_filename(output_filename)
                print(f"Uploaded {output_filename} to gs://{args.bucket}/{gcs_results_path}")
            except Exception as e:
                print(f"Failed to upload {output_filename} to GCS: {e}")

    print(f"Dataset {dataset_index} processing complete.")

if __name__ == '__main__':
    main()
