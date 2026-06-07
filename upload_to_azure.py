"""
Upload model outputs to Azure Blob Storage
"""
import os
from azure.storage.blob import BlobServiceClient
from pathlib import Path
from datetime import datetime

def upload_to_azure():
    """Upload model outputs to Azure Blob Storage"""
    connection_string = os.getenv('AZURE_STORAGE_CONNECTION_STRING')
    
    if not connection_string:
        print("❌ No Azure connection string found")
        print("Set AZURE_STORAGE_CONNECTION_STRING environment variable")
        return False
    
    try:
        blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        print("✅ Connected to Azure Blob Storage")
    except Exception as e:
        print(f"❌ Failed to connect to Azure: {e}")
        return False
    
    # Files to upload
    files_to_upload = {
        'predictions': [
            'data/ensemble_predictions.csv',  # Main predictions file (will be uploaded as ensemble_optimize_predictions.csv)
            'data/shap_values.csv',
            'data/ensemble_weights.csv'
        ],
        'reports': [
            'reports/oot_validation_report.csv',
            'reports/feature_ic_summary.csv',
            'reports/turnover_optimized.csv',
            'reports/comprehensive_diagnostic_report.txt'
        ]
    }
    
    timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
    uploaded_count = 0
    
    for container_name, files in files_to_upload.items():
        # Ensure container exists
        try:
            container_client = blob_service_client.get_container_client(container_name)
            if not container_client.exists():
                container_client.create_container(public_access='blob')
                print(f"✅ Created container: {container_name}")
        except Exception as e:
            print(f"⚠️  Container {container_name}: {e}")
        
        for file_path in files:
            if os.path.exists(file_path):
                try:
                    # Special handling for ensemble_predictions.csv
                    # Upload as ensemble_optimize_predictions.csv for backward compatibility
                    blob_name = os.path.basename(file_path)
                    if blob_name == 'ensemble_predictions.csv':
                        blob_name = 'ensemble_optimize_predictions.csv'
                    
                    blob_client = blob_service_client.get_blob_client(
                        container=container_name,
                        blob=blob_name
                    )
                    
                    with open(file_path, 'rb') as data:
                        blob_client.upload_blob(data, overwrite=True)
                    
                    print(f"✅ Uploaded {file_path} → {container_name}/{blob_name}")
                    uploaded_count += 1
                    
                    # Also upload with timestamp (archive)
                    archive_blob_name = f"archive/{timestamp}_{blob_name}"
                    archive_blob_client = blob_service_client.get_blob_client(
                        container=container_name,
                        blob=archive_blob_name
                    )
                    
                    with open(file_path, 'rb') as data:
                        archive_blob_client.upload_blob(data, overwrite=True)
                    
                    print(f"✅ Archived → {container_name}/{archive_blob_name}")
                    
                except Exception as e:
                    print(f"❌ Failed to upload {file_path}: {e}")
            else:
                print(f"⚠️  File not found: {file_path}")
    
    print(f"\n📊 Summary: Uploaded {uploaded_count} files to Azure Blob Storage")
    return uploaded_count > 0

if __name__ == "__main__":
    success = upload_to_azure()
    exit(0 if success else 1)
