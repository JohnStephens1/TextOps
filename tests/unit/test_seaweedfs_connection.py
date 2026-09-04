import boto3


def test_seaweedfs_connection() -> None:
    s3 = boto3.client(
        "s3",
        endpoint_url="http://seaweedfs:8333",
        aws_access_key_id="mlops-local",
        aws_secret_access_key="mlops-local-secret",
    )

    s3.list_buckets()
