"""Finalize phase: dataset manifest and a validation report for one run."""

import json
from dataclasses import asdict

import numpy as np
import pyarrow as pa

from atl06_processor.compact import FileStats
from atl06_processor.fragments import ConvertStats, write_parquet
from atl06_processor.paths import RunPaths


def _load(fs, directory: str, parse):
    """One batched ``cat`` (concurrent on s3fs) for every JSON in ``directory``."""
    paths = sorted(fs.glob(f"{directory}/*.json"))
    if not paths:
        return []
    contents = fs.cat(paths)
    return [parse(contents[p].decode()) for p in paths]


def _spread(values: list[int]) -> dict:
    if not values:
        return {"min": 0, "median": 0, "max": 0}
    a = np.asarray(values)
    return {"min": int(a.min()), "median": float(np.median(a)), "max": int(a.max())}


def finalize_run(fs, paths: RunPaths) -> dict:
    """Writes manifest/dataset.parquet and manifest/report.json; raises on row loss."""
    converts = _load(fs, paths.convert_stats_dir(), ConvertStats.from_json)
    files = _load(fs, paths.compact_stats_dir(), FileStats.from_json)

    rows_per_file = _spread([f.rows for f in files])
    rows_per_file["max_over_median"] = (
        rows_per_file["max"] / rows_per_file["median"] if rows_per_file["median"] else 0.0
    )
    fanout = np.asarray([c.ranges_touched for c in converts] or [0])
    report = {
        "granules": len(converts),
        "rows_in": sum(c.rows_in for c in converts),
        "rows_out_convert": sum(c.rows_out for c in converts),
        "rows_out_compact": sum(f.rows for f in files),
        "dropped_out_of_year": sum(c.dropped_out_of_year for c in converts),
        "dropped_invalid": sum(c.dropped_invalid for c in converts),
        "files": len(files),
        "rows_per_file": rows_per_file,
        "bytes_per_file": _spread([f.bytes for f in files]),
        "fanout": {
            "mean": float(fanout.mean()),
            "p50": float(np.percentile(fanout, 50)),
            "p95": float(np.percentile(fanout, 95)),
            "max": int(fanout.max()),
        },
    }
    if files:
        manifest = pa.Table.from_pylist([asdict(f) for f in files])
        write_parquet(fs, paths.dataset_manifest(), manifest)
    fs.pipe(paths.report(), json.dumps(report, indent=2).encode())
    if report["rows_out_convert"] != report["rows_out_compact"]:
        raise ValueError(
            f"row count mismatch: convert wrote {report['rows_out_convert']}, "
            f"compact wrote {report['rows_out_compact']}"
        )
    return report
