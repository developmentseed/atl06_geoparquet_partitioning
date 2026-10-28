from datetime import UTC, datetime

import fsspec
import numpy as np
import pyarrow.parquet as pq
import pytest
from atl06_processor.fragments import ConvertStats, convert_granule
from atl06_processor.granule import parse_granule_name
from atl06_processor.paths import RunPaths
from atl06_processor.splits import SplitTable

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
    src = make_granule(tmp_path / "src", NAME, beams)
    fs = fsspec.filesystem("file")
    paths = RunPaths(str(tmp_path / "run"))
    splits = SplitTable(np.array([2**60, 2**61, 2**62], np.uint64))

    def run():
        return convert_granule(fs, str(src), fs, paths, parse_granule_name(NAME), 3, 2023, splits)

    return fs, paths, splits, run


def fragment_files(fs, paths):
    return sorted(fs.glob(f"{paths.root}/staging/range=*/*.parquet"))


def test_convert_routes_rows_by_range(setup):
    fs, paths, splits, run = setup
    stats = run()
    total = 0
    for f in fragment_files(fs, paths):
        rid = int(f.split("range=")[1].split("/")[0])
        t = pq.read_table(f)
        assert set(splits.range_ids(t["stac_hash"].to_numpy())) == {rid}
        assert set(t["srcid"].to_pylist()) == {3}
        total += len(t)
    assert total == stats.rows_out == 1000


def test_convert_records_fanout(setup):
    fs, paths, _, run = setup
    stats = run()
    assert stats.ranges_touched == len(stats.fragment_rows) > 1
    saved = ConvertStats.from_json(fs.cat(paths.convert_stats(NAME)).decode())
    assert saved == stats


def test_convert_is_idempotent(setup):
    fs, paths, _, run = setup
    run()
    first = fragment_files(fs, paths)
    run()
    assert fragment_files(fs, paths) == first
    assert sum(pq.read_metadata(f).num_rows for f in first) == 1000
