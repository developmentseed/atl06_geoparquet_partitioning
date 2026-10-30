"""Deployment settings, read from ``CDK_*`` environment variables."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class StackSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="CDK_")

    account: str | None = None
    region: str = "us-west-2"
    bucket_name: str | None = None  # None: the stack creates (and retains) a bucket
    edl_secret_name: str = "atl06/earthdata"
    convert_concurrency: int = 50
    compact_concurrency: int = 100
    convert_batch_size: int = 1
    convert_memory_mb: int = 4096
    compact_memory_mb: int = 10240
    # Passed through to the processor's ATL06_* settings
    year: int = 2023
    target_rows_per_file: int = 10_000_000
    row_group_size: int = 500_000
    max_rows_per_file: int = 25_000_000
    sample_granule_fraction: float = 0.02
    sample_row_fraction: float = 0.01
