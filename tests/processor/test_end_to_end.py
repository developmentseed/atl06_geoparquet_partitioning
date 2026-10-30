"""Local, AWS-free run of every pipeline stage over synthetic granules."""

from datetime import UTC, datetime, timedelta

import duckdb
import numpy as np
import obstore
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from atl06_processor import storage
from atl06_processor.compact import compact_range
from atl06_processor.finalize import finalize_run
from atl06_processor.fragments import convert_granule
from atl06_processor.granule import parse_granule_name
from atl06_processor.paths import RunPaths
from atl06_processor.sample import sample_granule
from atl06_processor.splits import SplitTable
from obstore.store import LocalStore

from tests.conftest import BeamSpec, delta_time_for


def synthetic_granules(make_granule, directory, rng):
    """~20 granules over 2023 with points clustered at high latitudes, plus one at year end."""
    starts = [datetime(2023, 1, 1, tzinfo=UTC) + timedelta(days=18 * i) for i in range(20)]
    starts.append(datetime(2023, 12, 31, 23, 59, 50, tzinfo=UTC))
    paths = []
    for i, start in enumerate(starts):
        name = f"ATL06_{start:%Y%m%d%H%M%S}_{i:04d}0201_006_01.h5"
        beams = {}
        for gt in ("gt1l", "gt2l", "gt3l"):
            n = 400
            dt = delta_time_for(start) + np.arange(n) * 0.05
            lat = np.clip(rng.normal(-80 if i % 2 else 75, 5, n), -88, 88)
            beams[gt] = BeamSpec(dt, lat, rng.uniform(-180, 180, n))
        paths.append(make_granule(directory, name, beams))
    return paths


@pytest.fixture
def pipeline_run(make_granule, tmp_path):
    (tmp_path / "src").mkdir()
    srcs = synthetic_granules(make_granule, tmp_path / "src", np.random.default_rng(0))
    src = LocalStore(tmp_path / "src")
    store = LocalStore(tmp_path / "out", mkdir=True)
    paths = RunPaths("run")
    infos = [parse_granule_name(p.name) for p in srcs]

    samples = [sample_granule(src, p.name, info, 2023, 0.2, seed=i)
               for i, (p, info) in enumerate(zip(srcs, infos, strict=True))]
    total = sum(c for c, _ in samples)
    hashes = np.concatenate([h for _, h in samples])
    splits = SplitTable.from_sample(hashes, total, target_rows_per_file=total // 5)
    for srcid, (p, info) in enumerate(zip(srcs, infos, strict=True)):
        convert_granule(src, p.name, store, paths, info, srcid, 2023, splits)
    for rid, _, _ in splits.ranges():
        compact_range(store, paths, rid, row_group_size=1000, max_rows_per_file=10_000)
    root = tmp_path / "out"
    return store, root, paths, finalize_run(store, paths, root_uri=root.as_uri())


def read(store, key, **kwargs):
    return pq.read_table(pa.BufferReader(storage.read_bytes(store, key)), **kwargs)


def test_pipeline_end_to_end(pipeline_run):
    store, root, paths, report = pipeline_run
    assert report["rows_out_convert"] == report["rows_out_compact"]
    dropped = report["dropped_out_of_year"] + report["dropped_invalid"]
    assert report["rows_in"] == report["rows_out_convert"] + dropped
    assert report["dropped_out_of_year"] > 0
    assert report["files"] >= 4
    assert report["rows_per_file"]["max_over_median"] < 2
    for key in storage.list_keys(store, paths.data_dir(), ".parquet"):
        h = read(store, key, columns=["stac_hash"])["stac_hash"].to_numpy()
        assert np.all(np.diff(h.astype(np.int64)) >= 0)
    sql = f"SELECT count(*) FROM read_parquet('{root / paths.data_dir()}/*.parquet')"
    assert duckdb.sql(sql).fetchall()[0][0] == report["rows_out_compact"]


def test_dataset_manifest_lists_every_file(pipeline_run):
    store, _, paths, report = pipeline_run
    manifest = read(store, paths.dataset_manifest())
    assert manifest.num_rows == report["files"]
    assert sum(manifest["rows"].to_pylist()) == report["rows_out_compact"]


def test_finalize_detects_row_mismatch(pipeline_run):
    store, _, paths, _ = pipeline_run
    obstore.delete(store, storage.list_keys(store, paths.compact_stats_dir())[0])
    with pytest.raises(ValueError, match="row count mismatch"):
        finalize_run(store, paths)


@pytest.fixture
def read_calls(monkeypatch):
    """Counts storage reads: one get_many is a single concurrent batch."""
    calls = {"get_many": 0, "read_bytes": 0, "download_many": 0}
    for name in calls:
        original = getattr(storage, name)

        def counted(*args, _name=name, _original=original, **kwargs):
            calls[_name] += 1
            return _original(*args, **kwargs)

        monkeypatch.setattr(storage, name, counted)
    return calls


def test_finalize_batches_stats_reads(pipeline_run, read_calls):
    store, _, paths, _ = pipeline_run
    finalize_run(store, paths)
    assert read_calls == {"get_many": 2, "read_bytes": 0, "download_many": 0}


def test_compact_batches_fragment_reads(pipeline_run, read_calls):
    store, _, paths, _ = pipeline_run
    compact_range(store, paths, 0, row_group_size=1000, max_rows_per_file=10_000)
    assert read_calls == {"get_many": 0, "read_bytes": 0, "download_many": 1}
