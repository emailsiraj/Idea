import boto3
import json
import uuid
import cv2

from botocore.exceptions import ClientError
from io import BytesIO
from dotenv import load_dotenv


# ---------------------------------------------------
# 1. Create and return S3 client
# ---------------------------------------------------
load_dotenv()

BUCKET_NAME = "mset-user-details-files-bkt"
REGION_NAME = "us-east-1"

def get_s3_client(region_name=None):
    """
    Create and return an S3 client.

    Args:
        region_name (str | None): Optional AWS region

    Returns:
        boto3.client
    """
    return boto3.client("s3", 
                        region_name=region_name,
                        aws_access_key_id='',
                        aws_secret_access_key='')


# ---------------------------------------------------
# 2. Check if a folder exists in S3
# ---------------------------------------------------

def folder_exists(folder_path):
    """
    Check if a folder exists in an S3 bucket.

    Args:
        folder_path (str): e.g. "my/folder/"

    Returns:
        bool
    """
    s3 = get_s3_client()
    
    if not folder_path.endswith("/"):
        folder_path += "/"

    response = s3.list_objects_v2(
        Bucket=BUCKET_NAME,
        Prefix=folder_path,
        MaxKeys=1
    )

    return "Contents" in response


# ---------------------------------------------------
# 3. Download a file from S3
# ---------------------------------------------------

def download_file(key, local_path):
    """
    Download a file from S3.

    Args:
        key (str): full S3 object key (path + filename)
        local_path (str): where to save locally

    Returns:
        bool: True if success
    """
    try:
        s3 = get_s3_client()
        s3.download_file(BUCKET_NAME, key, local_path)
        return True
    except ClientError as e:
        print(f"Download failed: {e}")
        return False


# Optional: download into memory instead of disk
def download_file_bytes(key):
    """
    Download file and return bytes in memory.
    """
    buffer = BytesIO()
    try:
        s3 = get_s3_client()
        obj = s3.get_object(Bucket=BUCKET_NAME, Key=key)
        body = obj["Body"].read()
        return json.loads(body.decode("utf-8"))
    except ClientError as e:
        print(f"Download failed: {e}")
        return None


# ---------------------------------------------------
# 4. Upload JSON (overwrite if exists)
# ---------------------------------------------------

def upload_json(key, data_dict):
    """
    Upload a dictionary as JSON to S3.
    Overwrites if file exists.

    Args:
        key (str): full S3 key including filename.json
        data_dict (dict)

    Returns:
        bool
    """
    try:
        s3 = get_s3_client()
        json_bytes = json.dumps(data_dict, indent=2).encode("utf-8")

        s3.put_object(
            Bucket=BUCKET_NAME,
            Key=key,
            Body=json_bytes,
            ContentType="application/json"
        )

        return True

    except ClientError as e:
        print(f"Upload failed: {e}")
        return False

def upload_cv2_image_to_s3(
    img,
    folder_name: str,
    filename: str | None = None,
    jpeg_quality: int = 90,
    public: bool = True
) -> str:
    """
    Uploads an OpenCV image (BGR) to S3 as a JPEG and returns its URL.
    """

    if not folder_name.endswith("/"):
        folder_name += "/"

    if filename is None:
        filename = f"{uuid.uuid4().hex}.jpg"

    # Encode image to JPEG
    success, buffer = cv2.imencode(
        ".jpg",
        img,
        [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]
    )

    if not success:
        raise RuntimeError("Failed to encode image as JPEG")

    s3 = get_s3_client()

    extra_args = {
        "ContentType": "image/jpeg"
    }

    key = folder_name + filename

    s3.put_object(
        Bucket=BUCKET_NAME,
        Key=key,
        Body=buffer.tobytes(),
        **extra_args
    )

    return f"https://{BUCKET_NAME}.s3.{REGION_NAME}.amazonaws.com/{key}"

import boto3

def create_s3_folder(folder_name: str):
    """
    Creates a logical folder in S3 by creating an empty object
    with a trailing slash.
    """
    if not folder_name.endswith("/"):
        folder_name += "/"

    s3 = get_s3_client()
    s3.put_object(Bucket=BUCKET_NAME, Key=folder_name)
