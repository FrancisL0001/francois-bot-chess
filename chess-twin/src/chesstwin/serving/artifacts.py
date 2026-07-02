import boto3, os
from pathlib import Path

def fetch_model_from_r2(version: str, dest: Path) -> Path:
    """Download the versioned ONNX model from R2 to a local cache path."""
    if dest.exists():                      # already cached locally — skip download
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    client = boto3.client(
        "s3",
        endpoint_url=f"https://{os.environ['R2_ACCOUNT_ID']}.r2.cloudflarestorage.com",
        aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
    )
    client.download_file(
        os.environ["R2_BUCKET"],
        f"models/twin/{version}/twin.onnx",
        str(dest),
    )
    return dest