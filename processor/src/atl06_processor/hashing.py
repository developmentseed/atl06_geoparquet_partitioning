"""Filter rows to the target year and add the stac-hash sort key."""

from datetime import UTC, datetime

import pyarrow as pa
import pyarrow.compute as pc
from stac_hash import Hasher

from atl06_processor.schema import SCHEMA


def year_bounds(year: int) -> tuple[datetime, datetime]:
    """Half-open UTC interval [Jan 1 year, Jan 1 year+1)."""
    return datetime(year, 1, 1, tzinfo=UTC), datetime(year + 1, 1, 1, tzinfo=UTC)


def make_hasher(year: int) -> Hasher:
    # stac-hash treats `end` as inclusive; filter_and_hash enforces the half-open year.
    return Hasher(*year_bounds(year))


def filter_and_hash(
    batch: pa.RecordBatch, year: int, hasher: Hasher
) -> tuple[pa.RecordBatch, int, int]:
    """Drop unhashable rows, then rows outside the year (by their own timestamp), then
    append ``stac_hash``.

    Returns the hashed batch (conforming to SCHEMA), the number of rows dropped as out of
    year, and the number dropped as invalid (no time or no position, so no hash exists).
    """
    bbox = batch["bbox"]
    hashable = pc.and_(
        pc.is_valid(batch["time_ns"]),
        pc.and_(pc.is_valid(bbox.field("xmin")), pc.is_valid(bbox.field("ymin"))),
    )
    valid = batch.filter(hashable)
    start, end = year_bounds(year)
    t = valid["time_ns"]
    in_year = pc.and_(
        pc.greater_equal(t, pa.scalar(start.replace(tzinfo=None), t.type)),
        pc.less(t, pa.scalar(end.replace(tzinfo=None), t.type)),
    )
    kept = valid.filter(in_year)
    # stac-hash quantizes time far coarser than 1 us, so truncating ns is harmless.
    datetimes = kept["time_ns"].cast(pa.timestamp("us", tz="UTC"), safe=False).to_pylist()
    lon = kept["bbox"].field("xmin").to_pylist()
    lat = kept["bbox"].field("ymin").to_pylist()
    hashes = pa.array(hasher.hash_all(datetimes, lon, lat), pa.uint64())
    columns = [kept[f.name] if f.name != "stac_hash" else hashes for f in SCHEMA]
    out = pa.RecordBatch.from_arrays(columns, schema=SCHEMA)
    return out, len(valid) - len(kept), len(batch) - len(valid)
