from datetime import UTC, datetime

import h5py
import numpy as np
import pyarrow as pa
from atl06_processor.granule import parse_granule_name
from atl06_processor.hashing import filter_and_hash, make_hasher, year_bounds
from atl06_processor.reader import iter_beam_batches
from atl06_processor.schema import SCHEMA

from tests.conftest import BeamSpec, delta_time_for

NAME = "ATL06_20231231235959_00010301_006_01.h5"
TIMES = [
    datetime(2022, 12, 31, 23, 59, 59, tzinfo=UTC),
    datetime(2023, 6, 1, tzinfo=UTC),
    datetime(2024, 1, 1, tzinfo=UTC),
]


def batch(make_granule, tmp_path, times=TIMES):
    dt = np.array([delta_time_for(t) for t in times])
    n = len(times)
    path = make_granule(tmp_path, NAME, {"gt1l": BeamSpec(dt, np.full(n, -75.0), np.full(n, 20.0))})
    with h5py.File(path, "r") as h5:
        return next(iter_beam_batches(h5, parse_granule_name(NAME), 0))


def test_year_bounds():
    assert year_bounds(2023) == (datetime(2023, 1, 1, tzinfo=UTC), datetime(2024, 1, 1, tzinfo=UTC))


def test_filter_and_hash_drops_out_of_year(make_granule, tmp_path):
    out, dropped = filter_and_hash(batch(make_granule, tmp_path), 2023, make_hasher(2023))
    assert dropped == 2
    assert out["time_ns"].to_pylist() == [datetime(2023, 6, 1)]


def test_hash_matches_library(make_granule, tmp_path):
    hasher = make_hasher(2023)
    out, _ = filter_and_hash(batch(make_granule, tmp_path), 2023, hasher)
    assert out["stac_hash"][0].as_py() == hasher.hash(TIMES[1], 20.0, -75.0)


def test_output_conforms_to_schema(make_granule, tmp_path):
    out, _ = filter_and_hash(batch(make_granule, tmp_path), 2023, make_hasher(2023))
    assert out.schema.names == SCHEMA.names
    assert out.schema.field("stac_hash").type == pa.uint64()


def test_all_rows_out_of_year_returns_empty_batch(make_granule, tmp_path):
    b = batch(make_granule, tmp_path, [TIMES[0], TIMES[2]])
    out, dropped = filter_and_hash(b, 2023, make_hasher(2023))
    assert len(out) == 0 and dropped == 2 and out.schema.names == SCHEMA.names


def test_sub_microsecond_times_hash():
    t = pa.array([1_685_577_600_123_456_789], pa.timestamp("ns"))
    one = pa.array([1.0])
    b = pa.RecordBatch.from_arrays(
        [t, pa.StructArray.from_arrays([one, one, one, one], ["xmin", "ymin", "xmax", "ymax"])],
        names=["time_ns", "bbox"],
    )
    full = {f.name: pa.nulls(1, f.type) for f in SCHEMA if f.name not in ("time_ns", "bbox")}
    b = pa.RecordBatch.from_pydict(
        {**full, "time_ns": t, "bbox": b["bbox"]}, schema=pa.schema(
            [f for f in SCHEMA if f.name != "stac_hash"]))
    out, dropped = filter_and_hash(b, 2023, make_hasher(2023))
    assert dropped == 0 and out["stac_hash"][0].as_py() > 0
