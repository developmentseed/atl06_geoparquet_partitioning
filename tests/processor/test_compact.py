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
from atl06_processor.schema import OUTPUT_SCHEMA
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


def compact(store, paths, range_id=0, row_group_size=1000, max_rows_per_file=10_000):
    return compact_range(store, paths, range_id, row_group_size, max_rows_per_file)


def fragment_hashes(store, paths):
    keys = storage.list_keys(store, paths.fragment_dir(0), ".parquet")
    return np.sort(np.concatenate(
        [table(store, k, columns=["stac_hash"])["stac_hash"].to_numpy() for k in keys]))


def test_compact_sorts_by_hash(staged):
    store, paths = staged
    [stats] = compact(store, paths)
    h = table(store, stats.path)["stac_hash"].to_numpy()
    assert stats.rows == 2500 and np.all(np.diff(h.astype(np.int64)) >= 0)
    assert stats.hash_lo == int(h[0]) and stats.hash_hi == int(h[-1])


def test_single_part_when_under_limit(staged):
    store, paths = staged
    [stats] = compact(store, paths)
    assert (stats.range_id, stats.part) == (0, 0)
    assert stats.path == paths.data_file(0, 0) == "run/data/part-00000-00.parquet"
    assert storage.list_keys(store, paths.data_dir()) == [stats.path]


def test_split_parts_cover_every_row_once(staged):
    store, paths = staged
    parts = compact(store, paths, max_rows_per_file=1000)
    assert [p.part for p in parts] == [0, 1, 2]
    assert sum(p.rows for p in parts) == 2500
    assert all(p.rows <= 1000 for p in parts)
    for a, b in zip(parts, parts[1:], strict=False):
        assert a.hash_hi < b.hash_lo
    written = []
    for p in parts:
        h = table(store, p.path)["stac_hash"].to_numpy()
        assert np.all(np.diff(h.astype(np.int64)) >= 0) and len(h) == p.rows
        written.append(h)
    assert np.array_equal(np.concatenate(written), fragment_hashes(store, paths))
    keys = storage.list_keys(store, paths.compact_stats_dir(), ".json")
    assert keys == [paths.compact_stats(0, k) for k in range(3)]


def test_duplicate_hashes_stay_in_one_part(staged, monkeypatch):
    store, paths = staged
    # Every row shares one hash: no cut can separate them, so one part holds all rows.
    import atl06_processor.compact as compact_mod
    real = compact_mod._read_hashes
    monkeypatch.setattr(compact_mod, "_read_hashes",
                        lambda files: np.full(len(real(files)), 7, np.uint64))
    parts = compact(store, paths, max_rows_per_file=1000)
    assert [p.rows for p in parts] == [2500]


def test_output_files_use_output_schema(staged):
    store, paths = staged
    [stats] = compact(store, paths)
    t = table(store, stats.path)
    assert t.schema.equals(OUTPUT_SCHEMA, check_metadata=False)
    assert t["longitude"].to_pylist() == pc.struct_field(t["bbox"], "xmin").to_pylist()


def test_compact_row_groups(staged):
    store, paths = staged
    [stats] = compact(store, paths)
    assert meta(store, stats.path).num_row_groups == 3 == stats.row_groups


def test_compact_geo_metadata(staged):
    store, paths = staged
    [stats] = compact(store, paths)
    geo = json.loads(meta(store, stats.path).metadata[b"geo"])
    assert geo["columns"]["geometry"]["covering"]["bbox"]["ymax"] == ["bbox", "ymax"]
    assert geo["columns"]["geometry"]["bbox"] == [stats.xmin, stats.ymin, stats.xmax, stats.ymax]


def test_stats_exclude_fill_values(staged):
    store, paths = staged
    [stats] = compact(store, paths)
    md = meta(store, stats.path)
    col = md.schema.names.index("h_li")
    for i in range(md.num_row_groups):
        s = md.row_group(i).column(col).statistics
        assert s.has_min_max and s.max < 1e30 and s.null_count > 0


def test_bbox_and_lonlat_stats_present(staged):
    store, paths = staged
    [stats] = compact(store, paths)
    md = meta(store, stats.path)
    cols = [md.schema.column(i).path for i in range(md.num_columns)]
    for name in ("bbox.xmin", "longitude", "latitude", "time"):
        assert md.row_group(0).column(cols.index(name)).statistics.has_min_max


def test_compact_range_with_no_fragments_returns_empty(staged):
    store, paths = staged
    assert compact(store, paths, range_id=7) == []
    assert storage.list_keys(store, paths.data_dir()) == []


def test_file_stats_json_roundtrip(staged):
    store, paths = staged
    [stats] = compact(store, paths)
    saved = bytes(storage.read_bytes(store, paths.compact_stats(0, 0)))
    assert FileStats.from_json(saved) == stats
    t = table(store, stats.path)
    assert stats.time_min_ns == pc.min(t["time_ns"]).as_py()
    assert stats.time_max_ns == pc.max(t["time_ns"]).as_py()
    assert stats.xmin == pc.min(t["longitude"]).as_py()
    assert stats.nan_counts["longitude"] == 0 and "h_li" in stats.nan_counts
