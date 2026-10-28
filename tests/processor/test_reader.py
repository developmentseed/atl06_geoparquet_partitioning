from datetime import UTC, datetime

import h5py
import numpy as np
import pyarrow as pa
from atl06_processor.granule import GranuleInfo, parse_granule_name
from atl06_processor.reader import iter_beam_batches
from atl06_processor.schema import PRE_HASH_SCHEMA

from tests.conftest import FLOAT_FILL, BeamSpec, delta_time_for

NAME = "ATL06_20230105123456_01790201_006_01.h5"


def beam(n, when=datetime(2023, 1, 5, tzinfo=UTC), spot=1, **overrides):
    dt = delta_time_for(when) + np.arange(n, dtype=np.float64)
    return BeamSpec(dt, np.linspace(-70, -71, n), np.linspace(10, 11, n), spot, overrides)


def read(path, srcid=7):
    info = parse_granule_name(path.name)
    with h5py.File(path, "r") as h5:
        return list(iter_beam_batches(h5, info, srcid))


def test_parse_granule_name():
    assert parse_granule_name(NAME) == GranuleInfo(NAME, rgt=179, cycle=2, region=1)


def test_beam_batches_one_per_beam(make_granule, tmp_path):
    path = make_granule(tmp_path, NAME, {"gt1l": beam(5, spot=3), "gt2r": beam(5, spot=4)})
    batches = read(path)
    assert len(batches) == 2
    assert all(b.schema.equals(PRE_HASH_SCHEMA, check_metadata=True) for b in batches)
    table = pa.Table.from_batches(batches)
    assert set(table["gt"].to_pylist()) == {10, 40}
    assert sorted(set(table["spot"].to_pylist())) == [3, 4]
    assert set(table["srcid"].to_pylist()) == {7}
    assert set(table["rgt"].to_pylist()) == {179}
    assert set(table["cycle"].to_pylist()) == {2}
    assert set(table["region"].to_pylist()) == {1}
    assert table["segment_id"].to_pylist()[:2] == [1000, 1001]


def test_time_ns_is_utc(make_granule, tmp_path):
    when = datetime(2023, 3, 1, tzinfo=UTC)
    path = make_granule(tmp_path, NAME, {"gt1l": beam(1, when=when)})
    t = read(path)[0]["time_ns"][0].as_py()
    assert t.replace(tzinfo=UTC) == when


def test_fill_values_become_null(make_granule, tmp_path):
    h_li = np.array([1, 2, FLOAT_FILL, 4], np.float32)
    bsnow_conf = np.array([0, 127, 1, 2], np.int8)
    path = make_granule(tmp_path, NAME, {"gt1l": beam(4, h_li=h_li, bsnow_conf=bsnow_conf)})
    b = read(path)[0]
    assert b["h_li"].to_pylist() == [1.0, 2.0, None, 4.0]
    assert b["bsnow_conf"].to_pylist() == [0, None, 1, 2]


def test_missing_and_empty_beams_skipped(make_granule, tmp_path):
    path = make_granule(tmp_path, NAME, {"gt2l": beam(0), "gt3r": beam(3)})
    batches = read(path)
    assert len(batches) == 1
    assert set(batches[0]["gt"].to_pylist()) == {60}


def test_fill_lat_lon_time_become_null(make_granule, tmp_path):
    spec = beam(4)
    spec.lat = np.array([-70.0, 1.7976931348623157e308, -70.2, -70.3])
    spec.delta_time = spec.delta_time.copy()
    spec.delta_time[3] = 1.7976931348623157e308
    path = make_granule(tmp_path, NAME, {"gt1l": spec})
    b = read(path)[0]
    assert b["geometry"].null_count == 1 and not b["geometry"][1].is_valid
    assert b["bbox"].field("ymin")[1].as_py() is None
    assert b["time_ns"][3].as_py() is None and b["time_ns"][0].is_valid
