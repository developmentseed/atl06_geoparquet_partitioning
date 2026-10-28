"""NSIDC in-region S3 access with Earthdata Login, via obstore's credential provider."""

import json

from obstore.auth.earthdata import NasaEarthdataCredentialProvider
from obstore.store import S3Store

NSIDC_CREDENTIALS_URL = "https://data.nsidc.earthdatacloud.nasa.gov/s3credentials"


def earthdata_auth(secret_name: str) -> tuple[str, str]:
    """(username, password) from a Secrets Manager JSON secret."""
    import boto3

    secret = boto3.client("secretsmanager").get_secret_value(SecretId=secret_name)
    creds = json.loads(secret["SecretString"])
    return creds["username"], creds["password"]


def nsidc_store(
    bucket: str, secret_name: str, credentials_url: str = NSIDC_CREDENTIALS_URL
) -> S3Store:
    """The provider fetches temporary credentials (about 1 h) and refreshes them itself."""
    provider = NasaEarthdataCredentialProvider(credentials_url, auth=earthdata_auth(secret_name))
    return S3Store(bucket, region="us-west-2", credential_provider=provider)
