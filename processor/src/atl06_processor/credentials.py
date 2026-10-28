"""Temporary NSIDC S3 credentials (about 1 h lifetime), cached and refreshed."""

import json
import os
import time
from collections.abc import Callable

import s3fs

REFRESH_AFTER_S = 50 * 60


def fetch_nsidc_credentials(secret_name: str) -> dict:
    import boto3
    import earthaccess

    secret = boto3.client("secretsmanager").get_secret_value(SecretId=secret_name)
    creds = json.loads(secret["SecretString"])
    os.environ["EARTHDATA_USERNAME"] = creds["username"]
    os.environ["EARTHDATA_PASSWORD"] = creds["password"]
    earthaccess.login(strategy="environment")
    return earthaccess.get_s3_credentials(daac="NSIDC")


class NsidcS3:
    def __init__(
        self,
        secret_name: str,
        clock: Callable[[], float] = time.monotonic,
        fetch_credentials: Callable[[], dict] | None = None,
    ):
        self._clock = clock
        self._fetch = fetch_credentials or (lambda: fetch_nsidc_credentials(secret_name))
        self._fs: s3fs.S3FileSystem | None = None
        self._fetched_at = 0.0

    def fs(self) -> s3fs.S3FileSystem:
        if self._fs is None or self._clock() - self._fetched_at > REFRESH_AFTER_S:
            creds = self._fetch()
            self._fs = s3fs.S3FileSystem(
                key=creds["accessKeyId"],
                secret=creds["secretAccessKey"],
                token=creds["sessionToken"],
                # Granules are read whole: one GET beats many small HDF5 range reads.
                default_cache_type="all",
                skip_instance_cache=True,
            )
            self._fetched_at = self._clock()
        return self._fs
