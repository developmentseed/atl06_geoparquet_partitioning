"""Compact phase: merge one range's fragments into a single hash-sorted GeoParquet file."""

import json
from dataclasses import asdict, dataclass

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from atl06_processor.fragments import write_parquet
from atl06_processor.paths import RunPaths
from atl06_processor.schema import SCHEMA, geo_metadata


@dataclass
class FileStats:
    range_id: int
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


def _read_fragments(fs, files: list[str]) -> pa.Table:
    """One batched ``cat`` (concurrent on s3fs) for all fragments of a range."""
    contents = fs.cat(files)
    return pa.concat_tables(
        pq.read_table(pa.BufferReader(contents[p]), schema=SCHEMA) for p in files
    )


def compact_range(fs, paths: RunPaths, range_id: int, row_group_size: int) -> FileStats | None:
    """Returns None (writing nothing) when the range received no rows."""
    pattern = f"{paths.fragment_dir(range_id)}/*.parquet"
    files = sorted(fs.glob(pattern))
    if not files:
        return None
    table = _read_fragments(fs, files).sort_by("stac_hash")
    bbox = table["bbox"].combine_chunks()
    extent = (
        pc.min(bbox.field("xmin")).as_py(),
        pc.min(bbox.field("ymin")).as_py(),
        pc.max(bbox.field("xmax")).as_py(),
        pc.max(bbox.field("ymax")).as_py(),
    )
    metadata = {**(table.schema.metadata or {}), b"geo": json.dumps(geo_metadata(extent))}
    table = table.replace_schema_metadata(metadata)
    out = paths.data_file(range_id)
    write_parquet(fs, out, table, row_group_size=row_group_size, write_statistics=True)
    hashes, times = table["stac_hash"], table["time_ns"].cast(pa.int64())
    stats = FileStats(
        range_id=range_id,
        path=out,
        rows=len(table),
        bytes=fs.size(out),
        row_groups=-(-len(table) // row_group_size),
        hash_lo=hashes[0].as_py(),
        hash_hi=hashes[-1].as_py(),
        time_min_ns=pc.min(times).as_py(),
        time_max_ns=pc.max(times).as_py(),
        xmin=extent[0],
        ymin=extent[1],
        xmax=extent[2],
        ymax=extent[3],
    )
    fs.makedirs(paths.compact_stats_dir(), exist_ok=True)
    fs.pipe(paths.compact_stats(range_id), stats.to_json().encode())
    return stats
