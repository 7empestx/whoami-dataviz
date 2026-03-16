"""Stream repo files from S3 without local download."""
import boto3
from typing import Generator, List, Tuple

BUCKET = "nexio-code-kb-dev-720154970215"
PREFIX = "repos/"


def get_s3_client():
    """Return boto3 S3 client."""
    return boto3.client('s3')


def list_repos() -> List[str]:
    """List all repo names in the bucket."""
    s3 = get_s3_client()
    repos = []
    paginator = s3.get_paginator('list_objects_v2')

    for page in paginator.paginate(Bucket=BUCKET, Prefix=PREFIX, Delimiter='/'):
        for prefix in page.get('CommonPrefixes', []):
            # Extract repo name from 'repos/repo-name/'
            repo_name = prefix['Prefix'].replace(PREFIX, '').rstrip('/')
            repos.append(repo_name)

    return repos


def get_repo_files(repo_name: str) -> Generator[Tuple[str, str], None, None]:
    """Stream all files from a repo as (filename, content) tuples."""
    s3 = get_s3_client()
    repo_prefix = f"{PREFIX}{repo_name}/"
    paginator = s3.get_paginator('list_objects_v2')

    for page in paginator.paginate(Bucket=BUCKET, Prefix=repo_prefix):
        for obj in page.get('Contents', []):
            key = obj['Key']
            filename = key.replace(repo_prefix, '')

            # Skip directories and binary files
            if filename.endswith('/') or _is_binary_file(filename):
                continue

            try:
                file_obj = s3.get_object(Bucket=BUCKET, Key=key)
                content = file_obj['Body'].read().decode('utf-8', errors='ignore')
                yield (filename, content)
            except Exception:
                continue


def _is_binary_file(filename: str) -> bool:
    """Check if file is likely binary based on extension."""
    binary_extensions = {
        '.png', '.jpg', '.jpeg', '.gif', '.ico', '.pdf',
        '.zip', '.tar', '.gz', '.whl', '.pyc', '.so',
        '.exe', '.dll', '.bin', '.dat', '.pkl', '.npy',
        '.ttf', '.woff', '.woff2', '.eot', '.mp3', '.mp4',
        '.wav', '.avi', '.mov', '.webm', '.svg', '.lock'
    }
    return any(filename.lower().endswith(ext) for ext in binary_extensions)
