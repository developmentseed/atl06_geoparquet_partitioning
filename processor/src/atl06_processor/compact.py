"""Compact phase: merge one range's fragments into hash-sorted GeoParquet files.

Memory stays bounded however large a range is: fragments go to local disk, row counts come
from ``stac_hash`` alone, a range above ``max_rows_per_file`` is split at hash quantiles into
contiguous parts, and each part is sorted by index and written one row group at a time.
"""

import json
import math
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from atl06_processor import storage
from atl06_processor.paths import RunPaths
from atl06_processor.schema import OUTPUT_SCHEMA, SCHEMA, geo_metadata, to_output


@dataclass
class FileStats:
    range_id: int
    part: int
    path: str
    rows: int
    bytes: int
    row_groups: int
    hash_lo: int
    hash_hi: int
    time_min_ns: int
    time_max_ns: int
    xmin: float
    ymin: float
    xmax: float
    ymax: float

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, text: str) -> "FileStats":
        return cls(**json.loads(text))


def _read_hashes(files: list[Path]) -> np.ndarray:
    return np.concatenate(
        [pq.read_table(f, columns=["stac_hash"])["stac_hash"].to_numpy() for f in files]
    )


def _part_bounds(hashes: np.ndarray, max_rows: int) -> list[tuple[int | None, int | None]]:
    """Half-open ``[lo, hi)`` hash bounds per part (None = unbounded); equal hashes never split."""
    n = len(hashes)
    n_parts = max(1, math.ceil(n / max_rows))
    ordered = np.sort(hashes)
    cuts = np.unique(ordered[[k * n // n_parts for k in range(1, n_parts)]])
    cuts = cuts[cuts > ordered[0]]  # a cut at the minimum would leave the first part empty
    bounds = [None, *(int(c) for c in cuts), None]
    return list(zip(bounds[:-1], bounds[1:], strict=False))


def _part_filter(lo: int | None, hi: int | None):
    field, conditions = pc.field("stac_hash"), []
    if lo is not None:
        conditions.append(field >= pa.scalar(lo, pa.uint64()))
    if hi is not None:
        conditions.append(field < pa.scalar(hi, pa.uint64()))
    if not conditions:
        return None
    return conditions[0] if len(conditions) == 1 else conditions[0] & conditions[1]


def _write_part(
    store, paths: RunPaths, range_id: int, part: int, table: pa.Table, row_group_size: int
) -> FileStats:
    bbox = table["bbox"]
    extent = (
        pc.min(pc.struct_field(bbox, "xmin")).as_py(),
        pc.min(pc.struct_field(bbox, "ymin")).as_py(),
        pc.max(pc.struct_field(bbox, "xmax")).as_py(),
        pc.max(pc.struct_field(bbox, "ymax")).as_py(),
    )
    times = pc.cast(table["time_ns"], pa.int64())
    schema = OUTPUT_SCHEMA.with_metadata({b"geo": json.dumps(geo_metadata(extent))})
    order = pc.sort_indices(table["stac_hash"])
    out = paths.data_file(range_id, part)
    with storage.parquet_writer(
        store, out, schema, compression="zstd", write_statistics=True
    ) as writer:
        for start in range(0, len(order), row_group_size):
            rows = to_output(table.take(order[start : start + row_group_size]))
            writer.write_table(rows.replace_schema_metadata(schema.metadata),
                               row_group_size=row_group_size)
    stats = FileStats(
        range_id=range_id,
        part=part,
        path=out,
        rows=len(table),
        bytes=storage.size(store, out),
        row_groups=-(-len(table) // row_group_size),
        hash_lo=pc.min(table["stac_hash"]).as_py(),
        hash_hi=pc.max(table["stac_hash"]).as_py(),
        time_min_ns=pc.min(times).as_py(),
        time_max_ns=pc.max(times).as_py(),
        xmin=extent[0],
        ymin=extent[1],
        xmax=extent[2],
        ymax=extent[3],
    )
    storage.put_bytes(store, paths.compact_stats(range_id, part), stats.to_json().encode())
    return stats


def compact_range(
    store,
    paths: RunPaths,
    range_id: int,
    row_group_size: int,
    max_rows_per_file: int,
    workdir: Path | None = None,
) -> list[FileStats]:
    """One FileStats per written part; empty (writing nothing) when the range has no rows."""
    keys = storage.list_keys(store, paths.fragment_dir(range_id), ".parquet")
    if not keys:
        return []
    with tempfile.TemporaryDirectory(dir=workdir) as tmp:
        files = storage.download_many(store, keys, Path(tmp))
        dataset = ds.dataset(files, schema=SCHEMA, format="parquet")
        results = []
        for part, (lo, hi) in enumerate(_part_bounds(_read_hashes(files), max_rows_per_file)):
            table = dataset.to_table(filter=_part_filter(lo, hi))
            results.append(_write_part(store, paths, range_id, part, table, row_group_size))
            del table
        return results
