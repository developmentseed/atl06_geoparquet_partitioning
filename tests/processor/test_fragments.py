from datetime import UTC, datetime

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from atl06_processor import storage
from atl06_processor.fragments import ConvertStats, convert_granule
from atl06_processor.granule import parse_granule_name
from atl06_processor.paths import RunPaths
from atl06_processor.splits import SplitTable
from obstore.store import LocalStore

from tests.conftest import BeamSpec, delta_time_for

NAME = "ATL06_20230601000000_01790201_006_01.h5"


@pytest.fixture
def setup(make_granule, tmp_path):
    n = 500
    dt = delta_time_for(datetime(2023, 6, 1, tzinfo=UTC)) + np.arange(n) * 0.01
    beams = {
        "gt1l": BeamSpec(dt, np.linspace(-89, 89, n), np.linspace(-179, 179, n)),
        "gt2l": BeamSpec(dt, np.linspace(89, -89, n), np.linspace(179, -179, n)),
    }
    (tmp_path / "src").mkdir()
    make_granule(tmp_path / "src", NAME, beams)
    src = LocalStore(tmp_path / "src")
    out = LocalStore(tmp_path / "out", mkdir=True)
    paths = RunPaths("run")
    splits = SplitTable(np.array([2**60, 2**61, 2**62], np.uint64))

    def run():
        return convert_granule(src, NAME, out, paths, parse_granule_name(NAME), 3, 2023, splits)

    return out, paths, splits, run


def fragment_keys(store, paths):
    return storage.list_keys(store, f"{paths.root}/staging", ".parquet")


def read(store, key):
    return pq.read_table(pa.BufferReader(storage.read_bytes(store, key)))


def test_convert_routes_rows_by_range(setup):
    store, paths, splits, run = setup
    stats = run()
    total = 0
    for key in fragment_keys(store, paths):
        rid = int(key.split("range=")[1].split("/")[0])
        t = read(store, key)
        assert set(splits.range_ids(t["stac_hash"].to_numpy())) == {rid}
        assert set(t["srcid"].to_pylist()) == {3}
        total += len(t)
    assert total == stats.rows_out == 1000


def test_convert_records_fanout(setup):
    store, paths, _, run = setup
    stats = run()
    assert stats.ranges_touched == len(stats.fragment_rows) > 1
    saved = ConvertStats.from_json(bytes(storage.read_bytes(store, paths.convert_stats(NAME))))
    assert saved == stats


def test_convert_is_idempotent(setup):
    store, paths, _, run = setup
    run()
    first = fragment_keys(store, paths)
    run()
    assert fragment_keys(store, paths) == first
    assert sum(len(read(store, k)) for k in first) == 1000
