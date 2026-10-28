import json
from datetime import UTC, datetime

import fsspec
import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq
import pytest
from atl06_processor.compact import FileStats, compact_range
from atl06_processor.fragments import convert_granule
from atl06_processor.granule import parse_granule_name
from atl06_processor.paths import RunPaths
from atl06_processor.splits import SplitTable

from tests.conftest import FLOAT_FILL, BeamSpec, delta_time_for


@pytest.fixture
def staged(make_granule, tmp_path):
    """Three granules (2,500 rows) converted into a single range 0."""
    fs = fsspec.filesystem("file")
    paths = RunPaths(str(tmp_path / "run"))
    (tmp_path / "src").mkdir()
    splits = SplitTable(np.array([], np.uint64))
    rng = np.random.default_rng(0)
    for i, n in enumerate([1000, 1000, 500]):
        name = f"ATL06_2023060{i + 1}000000_0179020{i + 1}_006_01.h5"
        dt = delta_time_for(datetime(2023, 6, 1 + i, tzinfo=UTC)) + np.arange(n) * 0.01
        h_li = rng.uniform(0, 3000, n).astype(np.float32)
        h_li[::50] = FLOAT_FILL
        spec = BeamSpec(dt, rng.uniform(-90, -60, n), rng.uniform(-180, 180, n), overrides={"h_li": h_li})
        src = make_granule(tmp_path / "src", name, {"gt1l": spec})
        convert_granule(fs, str(src), fs, paths, parse_granule_name(name), i, 2023, splits)
    return fs, paths


def test_compact_sorts_by_hash(staged):
    fs, paths = staged
    stats = compact_range(fs, paths, 0, row_group_size=1000)
    h = pq.read_table(stats.path)["stac_hash"].to_numpy()
    assert stats.rows == 2500 and np.all(np.diff(h.astype(np.int64)) >= 0)
    assert stats.hash_lo == int(h[0]) and stats.hash_hi == int(h[-1])


def test_compact_row_groups(staged):
    fs, paths = staged
    stats = compact_range(fs, paths, 0, row_group_size=1000)
    assert pq.read_metadata(stats.path).num_row_groups == 3 == stats.row_groups


def test_compact_geo_metadata(staged):
    fs, paths = staged
    stats = compact_range(fs, paths, 0, row_group_size=1000)
    geo = json.loads(pq.read_schema(stats.path).metadata[b"geo"])
    assert geo["columns"]["geometry"]["covering"]["bbox"]["ymax"] == ["bbox", "ymax"]
    assert geo["columns"]["geometry"]["bbox"] == [stats.xmin, stats.ymin, stats.xmax, stats.ymax]


def test_stats_exclude_fill_values(staged):
    fs, paths = staged
    stats = compact_range(fs, paths, 0, row_group_size=1000)
    md = pq.read_metadata(stats.path)
    col = md.schema.names.index("h_li")
    for i in range(md.num_row_groups):
        s = md.row_group(i).column(col).statistics
        assert s.has_min_max and s.max < 1e30 and s.null_count > 0


def test_bbox_stats_present(staged):
    fs, paths = staged
    stats = compact_range(fs, paths, 0, row_group_size=1000)
    md = pq.read_metadata(stats.path)
    idx = [md.schema.column(i).path for i in range(md.num_columns)].index("bbox.xmin")
    assert md.row_group(0).column(idx).statistics.has_min_max


def test_compact_range_with_no_fragments_returns_none(staged):
    fs, paths = staged
    assert compact_range(fs, paths, 7, row_group_size=1000) is None
    assert not fs.exists(paths.data_file(7))


def test_file_stats_json_roundtrip(staged):
    fs, paths = staged
    stats = compact_range(fs, paths, 0, row_group_size=1000)
    assert FileStats.from_json(fs.cat(paths.compact_stats(0)).decode()) == stats
    t = pq.read_table(stats.path)
    assert stats.time_min_ns == pc.min(t["time_ns"]).value
