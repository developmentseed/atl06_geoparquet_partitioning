import json
from datetime import UTC, datetime

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import pytest
from atl06_processor import storage
from atl06_processor.compact import FileStats, compact_range
from atl06_processor.fragments import convert_granule
from atl06_processor.granule import parse_granule_name
from atl06_processor.paths import RunPaths
from atl06_processor.splits import SplitTable
from obstore.store import LocalStore

from tests.conftest import FLOAT_FILL, BeamSpec, delta_time_for


@pytest.fixture
def staged(make_granule, tmp_path):
    """Three granules (2,500 rows) converted into a single range 0."""
    store = LocalStore(tmp_path / "out", mkdir=True)
    paths = RunPaths("run")
    (tmp_path / "src").mkdir()
    src = LocalStore(tmp_path / "src")
    splits = SplitTable(np.array([], np.uint64))
    rng = np.random.default_rng(0)
    for i, n in enumerate([1000, 1000, 500]):
        name = f"ATL06_2023060{i + 1}000000_0179020{i + 1}_006_01.h5"
        dt = delta_time_for(datetime(2023, 6, 1 + i, tzinfo=UTC)) + np.arange(n) * 0.01
        h_li = rng.uniform(0, 3000, n).astype(np.float32)
        h_li[::50] = FLOAT_FILL
        lat, lon = rng.uniform(-90, -60, n), rng.uniform(-180, 180, n)
        spec = BeamSpec(dt, lat, lon, overrides={"h_li": h_li})
        make_granule(tmp_path / "src", name, {"gt1l": spec})
        convert_granule(src, name, store, paths, parse_granule_name(name), i, 2023, splits)
    return store, paths


def meta(store, key):
    return pq.read_metadata(pa.BufferReader(storage.read_bytes(store, key)))


def table(store, key, **kwargs):
    return pq.read_table(pa.BufferReader(storage.read_bytes(store, key)), **kwargs)


def test_compact_sorts_by_hash(staged):
    store, paths = staged
    stats = compact_range(store, paths, 0, row_group_size=1000)
    h = table(store, stats.path)["stac_hash"].to_numpy()
    assert stats.rows == 2500 and np.all(np.diff(h.astype(np.int64)) >= 0)
    assert stats.hash_lo == int(h[0]) and stats.hash_hi == int(h[-1])


def test_compact_row_groups(staged):
    store, paths = staged
    stats = compact_range(store, paths, 0, row_group_size=1000)
    assert meta(store, stats.path).num_row_groups == 3 == stats.row_groups


def test_compact_geo_metadata(staged):
    store, paths = staged
    stats = compact_range(store, paths, 0, row_group_size=1000)
    geo = json.loads(meta(store, stats.path).metadata[b"geo"])
    assert geo["columns"]["geometry"]["covering"]["bbox"]["ymax"] == ["bbox", "ymax"]
    assert geo["columns"]["geometry"]["bbox"] == [stats.xmin, stats.ymin, stats.xmax, stats.ymax]


def test_stats_exclude_fill_values(staged):
    store, paths = staged
    stats = compact_range(store, paths, 0, row_group_size=1000)
    md = meta(store, stats.path)
    col = md.schema.names.index("h_li")
    for i in range(md.num_row_groups):
        s = md.row_group(i).column(col).statistics
        assert s.has_min_max and s.max < 1e30 and s.null_count > 0


def test_bbox_stats_present(staged):
    store, paths = staged
    stats = compact_range(store, paths, 0, row_group_size=1000)
    md = meta(store, stats.path)
    idx = [md.schema.column(i).path for i in range(md.num_columns)].index("bbox.xmin")
    assert md.row_group(0).column(idx).statistics.has_min_max


def test_compact_range_with_no_fragments_returns_none(staged):
    store, paths = staged
    assert compact_range(store, paths, 7, row_group_size=1000) is None
    assert storage.list_keys(store, paths.data_dir()) == []


def test_file_stats_json_roundtrip(staged):
    store, paths = staged
    stats = compact_range(store, paths, 0, row_group_size=1000)
    saved = bytes(storage.read_bytes(store, paths.compact_stats(0)))
    assert FileStats.from_json(saved) == stats
    t = table(store, stats.path)
    assert stats.time_min_ns == pc.min(t["time_ns"]).value
