import os
import boto3
from botocore.config import Config
from botocore.exceptions import ClientError
import logging

# Configure silent logging for boto3 and botocore
logging.getLogger('boto3').setLevel(logging.CRITICAL)
logging.getLogger('botocore').setLevel(logging.CRITICAL)
logging.getLogger('s3transfer').setLevel(logging.CRITICAL)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def _make_s3_client():
    access_key = os.environ.get('AWS_ACCESS_KEY_ID')
    secret_key = os.environ.get('AWS_SECRET_ACCESS_KEY')
    region = os.environ.get('AWS_REGION', 'us-east-1')
    endpoint_url = os.environ.get('AWS_S3_ENDPOINT_URL') or os.environ.get('S3_ENDPOINT_URL')

    if not access_key or not secret_key:
        return None

    force_path_style = os.environ.get('AWS_S3_FORCE_PATH_STYLE', '').strip().lower() in {'1', 'true', 'yes'}
    kwargs = {
        'aws_access_key_id': access_key,
        'aws_secret_access_key': secret_key,
        'region_name': region,
    }
    if endpoint_url:
        kwargs['endpoint_url'] = endpoint_url
    if force_path_style or endpoint_url:
        kwargs['config'] = Config(s3={'addressing_style': 'path'})
    return boto3.client('s3', **kwargs)

def upload_file_to_s3(file_path, bucket_name, s3_key):
    """
    Upload a file to an S3 bucket silently.
    """
    s3_client = _make_s3_client()
    if s3_client is None:
        return False
    try:
        # Extra arguments for public read if needed, but the user didn't specify.
        # Given the bucket name, it might be for a web app.
        s3_client.upload_file(file_path, bucket_name, s3_key)
        return True
    except ClientError:
        return False
    except Exception:
        return False

def upload_job_artifacts(directory, job_id):
    """
    Upload all generated clips and metadata for a job to S3.
    """
    bucket_name = os.environ.get('AWS_S3_BUCKET', 'openshorts.app-clips')
    
    if not os.path.exists(directory):
        return

    for filename in os.listdir(directory):
        # Upload .mp4 clips and the metadata JSON
        if (filename.endswith(".mp4") or filename.endswith(".json")) and not filename.startswith("temp_"):
            file_path = os.path.join(directory, filename)
            s3_key = f"{job_id}/{filename}"
            upload_file_to_s3(file_path, bucket_name, s3_key)

def list_remote_job_ids(limit=200):
    """
    List job prefixes available in S3/MinIO without downloading artifacts.
    """
    bucket_name = os.environ.get('AWS_S3_BUCKET', 'openshorts.app-clips')
    s3_client = _make_s3_client()
    if s3_client is None:
        return []

    out = []
    seen = set()
    try:
        paginator = s3_client.get_paginator('list_objects_v2')
        for page in paginator.paginate(Bucket=bucket_name, Delimiter='/'):
            for prefix_item in page.get('CommonPrefixes', []):
                prefix = str(prefix_item.get('Prefix') or '').strip('/')
                safe_job_id = os.path.basename(prefix)
                if safe_job_id and safe_job_id not in seen:
                    seen.add(safe_job_id)
                    out.append(safe_job_id)
                    if len(out) >= int(limit or 200):
                        return out

            for item in page.get('Contents', []):
                key = str(item.get('Key') or '')
                safe_job_id = os.path.basename(key.split('/', 1)[0])
                if safe_job_id and safe_job_id not in seen:
                    seen.add(safe_job_id)
                    out.append(safe_job_id)
                    if len(out) >= int(limit or 200):
                        return out
    except Exception as exc:
        logger.warning("Failed to list S3 job prefixes: %s", exc)
    return out

def download_job_artifacts(job_id, destination_dir):
    """
    Download all generated clips and metadata for a job from S3/MinIO.
    """
    safe_job_id = os.path.basename(str(job_id or "").strip())
    if not safe_job_id:
        return 0

    bucket_name = os.environ.get('AWS_S3_BUCKET', 'openshorts.app-clips')
    s3_client = _make_s3_client()
    if s3_client is None:
        return 0

    os.makedirs(destination_dir, exist_ok=True)
    downloaded = 0
    try:
        paginator = s3_client.get_paginator('list_objects_v2')
        for page in paginator.paginate(Bucket=bucket_name, Prefix=f"{safe_job_id}/"):
            for item in page.get('Contents', []):
                key = item.get('Key')
                if not key:
                    continue
                filename = os.path.basename(str(key))
                if not filename or filename.startswith("temp_"):
                    continue
                if not (filename.endswith(".mp4") or filename.endswith(".json")):
                    continue
                local_path = os.path.join(destination_dir, filename)
                s3_client.download_file(bucket_name, key, local_path)
                downloaded += 1
    except Exception as exc:
        logger.warning("Failed to download S3 artifacts for %s: %s", safe_job_id, exc)
    return downloaded

def delete_job_artifacts(job_id):
    """
    Delete all generated clips and metadata for a job from S3/MinIO.
    """
    safe_job_id = os.path.basename(str(job_id or "").strip())
    if not safe_job_id:
        return 0
    bucket_name = os.environ.get('AWS_S3_BUCKET', 'openshorts.app-clips')
    s3_client = _make_s3_client()
    if s3_client is None:
        return 0

    deleted = 0
    try:
        paginator = s3_client.get_paginator('list_objects_v2')
        for page in paginator.paginate(Bucket=bucket_name, Prefix=f"{safe_job_id}/"):
            for item in page.get('Contents', []):
                key = item.get('Key')
                if not key:
                    continue
                s3_client.delete_object(Bucket=bucket_name, Key=key)
                deleted += 1
    except Exception as exc:
        logger.warning("Failed to delete S3 artifacts for %s: %s", safe_job_id, exc)
    return deleted
