"""AWS Lambda entry points; thin wrappers over the library.

Map steps receive ``{"Items": [...], "BatchInput": {"run_prefix": ...}}``;
other steps receive the execution input ``{"run_prefix", "start"?, "end"?, "limit"?}``.
"""

import os
from datetime import datetime
from functools import cache

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from obstore.store import S3Store

from atl06_processor import storage
from atl06_processor.compact import compact_range
from atl06_processor.config import Settings
from atl06_processor.credentials import nsidc_store
from atl06_processor.finalize import finalize_run
from atl06_processor.fragments import convert_granule
from atl06_processor.granule import parse_granule_name
from atl06_processor.hashing import year_bounds
from atl06_processor.inventory import build_inventory, search_granules
from atl06_processor.paths import RunPaths
from atl06_processor.sample import sample_granule
from atl06_processor.splits import SplitTable


def get_output_store() -> S3Store:
    """The run bucket, using the Lambda role's credentials from the environment."""
    return S3Store(Settings().bucket, region=os.environ.get("AWS_REGION", "us-west-2"))


def _nsidc_store(bucket: str) -> S3Store:
    return nsidc_store(bucket, Settings().edl_secret_name)


@cache
def _source_store(bucket: str) -> S3Store:
    # One store (and one Earthdata credential provider) per bucket per container.
    return _nsidc_store(bucket)


def get_source(s3_url: str) -> tuple[S3Store, str]:
    bucket, key = storage.split_s3_url(s3_url)
    return _source_store(bucket), key


def _run_prefix(event: dict, settings: Settings) -> str:
    return (
        event.get("run_prefix")
        or event.get("BatchInput", {}).get("run_prefix")
        or settings.run_prefix
    )


def _context(event: dict) -> tuple[Settings, RunPaths, str]:
    settings = Settings()
    prefix = _run_prefix(event, settings)
    return settings, RunPaths(prefix), prefix


def inventory_handler(event: dict, context=None) -> dict:
    settings, paths, prefix = _context(event)
    start, end = year_bounds(settings.year)
    if event.get("start"):
        start = datetime.fromisoformat(event["start"])
    if event.get("end"):
        end = datetime.fromisoformat(event["end"])
    year_start, year_end = year_bounds(settings.year)
    if start < year_start or end > year_end or start >= end:
        # Points are filtered to ATL06_YEAR, so anything outside it would silently vanish.
        raise ValueError(
            f"search window {start.isoformat()}..{end.isoformat()} is outside ATL06_YEAR "
            f"{settings.year} (or empty)"
        )
    store = get_output_store()
    # A prefix is single-use: stale fragments routed under old cuts would corrupt the output.
    if storage.prefix_exists(store, paths.root):
        if not event.get("overwrite"):
            raise ValueError(
                f"run prefix {prefix!r} already exists; use a new run_prefix or pass "
                '"overwrite": true to delete it first'
            )
        storage.delete_prefix(store, paths.root)
    granules = search_granules(settings, start, end)
    if not granules:
        # Fail here, not three steps later: an empty inventory makes every later step vacuous.
        raise ValueError(
            f"no {settings.short_name} v{settings.version} granules found in CMR for "
            f"{start.isoformat()}..{end.isoformat()}; check ATL06_VERSION is still published"
        )
    table, items, sampled = build_inventory(
        granules, settings.sample_granule_fraction, seed=0, limit=event.get("limit")
    )
    storage.put_parquet(store, paths.granules_table(), table)
    storage.put_json(store, paths.granules_json(), items)
    storage.put_json(store, paths.sample_json(), sampled)
    return {
        "run_prefix": prefix,
        "bucket": settings.bucket,
        "granules": len(items),
        "sampled": len(sampled),
        "granules_key": paths.granules_json(),
        "sample_key": paths.sample_json(),
    }


def sample_handler(event: dict, context=None) -> dict:
    settings, paths, _ = _context(event)
    store = get_output_store()
    for item in event["Items"]:
        info = parse_granule_name(item["granule"])
        src_store, src_key = get_source(item["s3_url"])
        count, hashes = sample_granule(
            src_store, src_key, info, settings.year, settings.sample_row_fraction,
            seed=item["srcid"],
        )
        table = pa.table({"stac_hash": pa.array(hashes, pa.uint64())})
        table = table.replace_schema_metadata({"row_count": str(count)})
        storage.put_parquet(store, paths.sample_part(info.name), table)
    return {"sampled": len(event["Items"])}


def plan_splits_handler(event: dict, context=None) -> dict:
    settings, paths, prefix = _context(event)
    store = get_output_store()
    total_granules = len(storage.get_json(store, paths.granules_json()))
    sampled_granules = len(storage.get_json(store, paths.sample_json()))
    rows, hashes = 0, []
    parts = storage.list_keys(store, paths.sample_dir(), ".parquet")
    contents = storage.get_many(store, parts)
    for part in parts:
        table = pq.read_table(pa.BufferReader(contents[part]))
        rows += int(table.schema.metadata[b"row_count"])
        hashes.append(table["stac_hash"].to_numpy())
    # Use the realized sample fraction, which is exact after rounding and limits.
    est_total_rows = round(rows * total_granules / sampled_granules)
    sample = np.concatenate(hashes) if hashes else np.array([], np.uint64)
    splits = SplitTable.from_sample(sample, est_total_rows, settings.target_rows_per_file)
    splits.save(store, paths.splits())
    storage.put_json(store, paths.ranges_json(), [{"range_id": r} for r, _, _ in splits.ranges()])
    return {
        "ranges": splits.n_ranges,
        "est_total_rows": est_total_rows,
        "ranges_key": paths.ranges_json(),
    }


def convert_handler(event: dict, context=None) -> list[dict]:
    settings, paths, _ = _context(event)
    store = get_output_store()
    splits = SplitTable.load(store, paths.splits())
    results = []
    for item in event["Items"]:
        info = parse_granule_name(item["granule"])
        src_store, src_key = get_source(item["s3_url"])
        stats = convert_granule(
            src_store, src_key, store, paths, info, item["srcid"], settings.year, splits
        )
        results.append(
            {"granule": info.name, "rows_out": stats.rows_out,
             "ranges_touched": stats.ranges_touched}
        )
    return results


def compact_handler(event: dict, context=None) -> list[dict]:
    settings, paths, _ = _context(event)
    store = get_output_store()
    results = []
    for item in event["Items"]:
        stats = compact_range(store, paths, item["range_id"], settings.row_group_size)
        results.append({"range_id": item["range_id"], "rows": stats.rows if stats else 0})
    return results


def finalize_handler(event: dict, context=None) -> dict:
    _, paths, _ = _context(event)
    return finalize_run(get_output_store(), paths)
