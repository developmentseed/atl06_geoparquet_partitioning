"""AWS Lambda entry points; thin wrappers over the library.

Map steps receive ``{"Items": [...], "BatchInput": {"run_prefix": ...}}``;
other steps receive the execution input ``{"run_prefix", "start"?, "end"?, "limit"?}``.
"""

import json
from datetime import datetime
from functools import cache

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import s3fs

from atl06_processor.compact import compact_range
from atl06_processor.config import Settings
from atl06_processor.credentials import NsidcS3
from atl06_processor.finalize import finalize_run
from atl06_processor.fragments import convert_granule, write_parquet
from atl06_processor.granule import parse_granule_name
from atl06_processor.hashing import year_bounds
from atl06_processor.inventory import build_inventory, search_granules
from atl06_processor.paths import RunPaths
from atl06_processor.sample import sample_granule
from atl06_processor.splits import SplitTable


@cache
def _nsidc() -> NsidcS3:
    return NsidcS3(Settings().edl_secret_name)


def get_output_fs():
    return s3fs.S3FileSystem()


def get_source_fs():
    return _nsidc().fs()


def _run_prefix(event: dict, settings: Settings) -> str:
    return (
        event.get("run_prefix")
        or event.get("BatchInput", {}).get("run_prefix")
        or settings.run_prefix
    )


def _context(event: dict) -> tuple[Settings, RunPaths, str]:
    settings = Settings()
    prefix = _run_prefix(event, settings)
    return settings, RunPaths(f"{settings.bucket}/{prefix}"), prefix


def _put_json(fs, path: str, obj) -> None:
    fs.makedirs(path.rsplit("/", 1)[0], exist_ok=True)
    fs.pipe(path, json.dumps(obj).encode())


def inventory_handler(event: dict, context=None) -> dict:
    settings, paths, prefix = _context(event)
    start, end = year_bounds(settings.year)
    if event.get("start"):
        start = datetime.fromisoformat(event["start"])
    if event.get("end"):
        end = datetime.fromisoformat(event["end"])
    fs = get_output_fs()
    # A prefix is single-use: stale fragments routed under old cuts would corrupt the output.
    if fs.exists(paths.root) and fs.ls(paths.root):
        if not event.get("overwrite"):
            raise ValueError(
                f"run prefix {prefix!r} already exists; use a new run_prefix or pass "
                '"overwrite": true to delete it first'
            )
        fs.rm(paths.root, recursive=True)
    granules = search_granules(settings, start, end)
    table, items, sampled = build_inventory(
        granules, settings.sample_granule_fraction, seed=0, limit=event.get("limit")
    )
    write_parquet(fs, paths.granules_table(), table)
    _put_json(fs, paths.granules_json(), items)
    _put_json(fs, paths.sample_json(), sampled)
    return {
        "run_prefix": prefix,
        "bucket": settings.bucket,
        "granules": len(items),
        "sampled": len(sampled),
        "granules_key": f"{prefix}/manifest/granules.json",
        "sample_key": f"{prefix}/manifest/sample.json",
    }


def sample_handler(event: dict, context=None) -> dict:
    settings, paths, _ = _context(event)
    out_fs, src_fs = get_output_fs(), get_source_fs()
    for item in event["Items"]:
        info = parse_granule_name(item["granule"])
        count, hashes = sample_granule(
            src_fs, item["s3_url"], info, settings.year, settings.sample_row_fraction,
            seed=item["srcid"],
        )
        table = pa.table({"stac_hash": pa.array(hashes, pa.uint64())})
        table = table.replace_schema_metadata({"row_count": str(count)})
        write_parquet(out_fs, paths.sample_part(info.name), table)
    return {"sampled": len(event["Items"])}


def plan_splits_handler(event: dict, context=None) -> dict:
    settings, paths, prefix = _context(event)
    fs = get_output_fs()
    total_granules = len(json.loads(fs.cat(paths.granules_json())))
    sampled_granules = len(json.loads(fs.cat(paths.sample_json())))
    rows, hashes = 0, []
    for part in sorted(fs.glob(f"{paths.sample_dir()}/*.parquet")):
        with fs.open(part, "rb") as f:
            table = pq.read_table(f)
        rows += int(table.schema.metadata[b"row_count"])
        hashes.append(table["stac_hash"].to_numpy())
    # Use the realized sample fraction, which is exact after rounding and limits.
    est_total_rows = round(rows * total_granules / sampled_granules)
    sample = np.concatenate(hashes) if hashes else np.array([], np.uint64)
    splits = SplitTable.from_sample(sample, est_total_rows, settings.target_rows_per_file)
    splits.save(fs, paths.splits())
    _put_json(fs, paths.ranges_json(), [{"range_id": r} for r, _, _ in splits.ranges()])
    return {
        "ranges": splits.n_ranges,
        "est_total_rows": est_total_rows,
        "ranges_key": f"{prefix}/manifest/ranges.json",
    }


def convert_handler(event: dict, context=None) -> list[dict]:
    settings, paths, _ = _context(event)
    out_fs, src_fs = get_output_fs(), get_source_fs()
    splits = SplitTable.load(out_fs, paths.splits())
    results = []
    for item in event["Items"]:
        info = parse_granule_name(item["granule"])
        stats = convert_granule(
            src_fs, item["s3_url"], out_fs, paths, info, item["srcid"], settings.year, splits
        )
        results.append(
            {"granule": info.name, "rows_out": stats.rows_out,
             "ranges_touched": stats.ranges_touched}
        )
    return results


def compact_handler(event: dict, context=None) -> list[dict]:
    settings, paths, _ = _context(event)
    fs = get_output_fs()
    results = []
    for item in event["Items"]:
        stats = compact_range(fs, paths, item["range_id"], settings.row_group_size)
        results.append({"range_id": item["range_id"], "rows": stats.rows if stats else 0})
    return results


def finalize_handler(event: dict, context=None) -> dict:
    _, paths, _ = _context(event)
    return finalize_run(get_output_fs(), paths)
