"""Convert phase: stream one granule and route its rows into per-range staging fragments."""

import json
from dataclasses import asdict, dataclass, field

import h5py
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from atl06_processor.granule import GranuleInfo
from atl06_processor.hashing import filter_and_hash, make_hasher
from atl06_processor.paths import RunPaths
from atl06_processor.reader import iter_beam_batches
from atl06_processor.schema import SCHEMA
from atl06_processor.splits import SplitTable


@dataclass
class ConvertStats:
    granule: str
    rows_in: int = 0
    rows_out: int = 0
    dropped_out_of_year: int = 0
    dropped_invalid: int = 0
    ranges_touched: int = 0
    fragment_rows: dict[int, int] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, text: str) -> "ConvertStats":
        d = json.loads(text)
        d["fragment_rows"] = {int(k): v for k, v in d["fragment_rows"].items()}
        return cls(**d)


def write_parquet(fs, path: str, table: pa.Table, **kwargs) -> None:
    fs.makedirs(path.rsplit("/", 1)[0], exist_ok=True)
    with fs.open(path, "wb") as f:
        pq.write_table(table, f, compression="zstd", **kwargs)


def convert_granule(
    src_fs,
    src_path: str,
    out_fs,
    paths: RunPaths,
    info: GranuleInfo,
    srcid: int,
    year: int,
    splits: SplitTable,
) -> ConvertStats:
    """Fragment keys are deterministic per (range, granule), so retries overwrite."""
    hasher = make_hasher(year)
    stats = ConvertStats(granule=info.name)
    by_range: dict[int, list[pa.RecordBatch]] = {}
    with src_fs.open(src_path, "rb") as f, h5py.File(f, "r") as h5:
        for batch in iter_beam_batches(h5, info, srcid):
            hashed, out_of_year, invalid = filter_and_hash(batch, year, hasher)
            stats.rows_in += len(batch)
            stats.dropped_out_of_year += out_of_year
            stats.dropped_invalid += invalid
            rids = splits.range_ids(hashed["stac_hash"].to_numpy(zero_copy_only=False))
            for rid in np.unique(rids):
                by_range.setdefault(int(rid), []).append(
                    hashed.filter(pc.equal(pa.array(rids), rid))
                )
    for rid, batches in sorted(by_range.items()):
        table = pa.Table.from_batches(batches, schema=SCHEMA)
        write_parquet(out_fs, paths.fragment(rid, info.name), table)
        stats.fragment_rows[rid] = len(table)
    stats.rows_out = sum(stats.fragment_rows.values())
    stats.ranges_touched = len(stats.fragment_rows)
    out_fs.makedirs(paths.convert_stats_dir(), exist_ok=True)
    out_fs.pipe(paths.convert_stats(info.name), stats.to_json().encode())
    return stats
