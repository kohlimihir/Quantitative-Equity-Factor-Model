import logging
import azure.functions as func
from datetime import datetime
import subprocess
import os
from azure.storage.blob import BlobServiceClient

def main(mytimer: func.TimerRequest) -> None:
    """
    Azure Function that runs monthly on the 1st day
    Triggers model retraining and uploads results to Blob Storage
    """
    utc_timestamp = datetime.utcnow().isoformat()
    
    if mytimer.past_due:
        logging.info('The timer is past due!')
    
    logging.info(f'Monthly model retrain started at: {utc_timestamp}')
    
    try:
        # Run the model pipeline
        logging.info('Starting model training...')
        result = subprocess.run(
            ['python', 'run_all.py', '--config', 'optimized_low_turnover'],
            capture_output=True,
            text=True,
            timeout=3600  # 1 hour timeout
        )
        
        if result.returncode != 0:
            logging.error(f'Model training failed: {result.stderr}')
            raise Exception(f'Model training failed: {result.stderr}')
        
        logging.info('Model training completed successfully')
        
        # Upload results to Azure Blob Storage
        connection_string = os.getenv('AZURE_STORAGE_CONNECTION_STRING')
        
        if connection_string:
            logging.info('Uploading results to Blob Storage...')
            upload_results_to_blob(connection_string)
            logging.info('Results uploaded successfully')
        else:
            logging.warning('No Azure Storage connection string found')
        
        logging.info(f'Monthly model retrain completed at: {datetime.utcnow().isoformat()}')
    
    except Exception as e:
        logging.error(f'Error during model retrain: {str(e)}')
        raise

def upload_results_to_blob(connection_string: str):
    """Upload model outputs to Azure Blob Storage"""
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    container_name = "model-outputs"
    
    # Create container if it doesn't exist
    try:
        blob_service_client.create_container(container_name)
    except:
        pass  # Container already exists
    
    # Files to upload
    files_to_upload = [
        'data/ensemble_optimize_predictions.csv',
        'data/shap_values.csv',
        'reports/oot_validation_report.csv',
        'reports/turnover_optimized.csv',
        'reports/feature_ic_summary.csv',
        'reports/comprehensive_diagnostic_report.txt'
    ]
    
    timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
    
    for file_path in files_to_upload:
        if os.path.exists(file_path):
            blob_name = f"{timestamp}/{file_path}"
            blob_client = blob_service_client.get_blob_client(
                container=container_name,
                blob=blob_name
            )
            
            with open(file_path, 'rb') as data:
                blob_client.upload_blob(data, overwrite=True)
            
            logging.info(f'Uploaded {file_path} to {blob_name}')
        else:
            logging.warning(f'File not found: {file_path}')
