"""Runtime settings, read from ``ATL06_*`` environment variables."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="ATL06_")

    year: int = 2023
    short_name: str = "ATL06"
    version: str = "007"  # v006 was retired from CMR; 007 is the only ATL06 release
    target_rows_per_file: int = 10_000_000
    row_group_size: int = 500_000
    # Compact splits a range above this into several files, bounding Lambda memory.
    max_rows_per_file: int = 25_000_000  # ~195 B/row Arrow peak measured → ~5 GB of 10 GB
    sample_granule_fraction: float = 0.02
    sample_row_fraction: float = 0.01
    bucket: str = ""
    run_prefix: str = "runs/dev"
    edl_secret_name: str = "atl06/earthdata"
