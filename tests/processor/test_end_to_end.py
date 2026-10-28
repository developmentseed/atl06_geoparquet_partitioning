"""Local, AWS-free run of every pipeline stage over synthetic granules."""

from datetime import UTC, datetime, timedelta

import duckdb
import fsspec
import fsspec.implementations.local
import numpy as np
import pyarrow.parquet as pq
import pytest
from atl06_processor.compact import compact_range
from atl06_processor.finalize import finalize_run
from atl06_processor.fragments import convert_granule
from atl06_processor.granule import parse_granule_name
from atl06_processor.paths import RunPaths
from atl06_processor.sample import sample_granule
from atl06_processor.splits import SplitTable

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
    fs = fsspec.filesystem("file")
    (tmp_path / "src").mkdir()
    srcs = synthetic_granules(make_granule, tmp_path / "src", np.random.default_rng(0))
    paths = RunPaths(str(tmp_path / "run"))
    infos = [parse_granule_name(p.name) for p in srcs]

    samples = [sample_granule(fs, str(p), info, 2023, 0.2, seed=i)
               for i, (p, info) in enumerate(zip(srcs, infos, strict=True))]
    total = sum(c for c, _ in samples)
    hashes = np.concatenate([h for _, h in samples])
    splits = SplitTable.from_sample(hashes, total, target_rows_per_file=total // 5)
    for srcid, (p, info) in enumerate(zip(srcs, infos, strict=True)):
        convert_granule(fs, str(p), fs, paths, info, srcid, 2023, splits)
    for rid, _, _ in splits.ranges():
        compact_range(fs, paths, rid, row_group_size=1000)
    return fs, paths, finalize_run(fs, paths)


def test_pipeline_end_to_end(pipeline_run):
    fs, paths, report = pipeline_run
    assert report["rows_out_convert"] == report["rows_out_compact"]
    dropped = report["dropped_out_of_year"] + report["dropped_invalid"]
    assert report["rows_in"] == report["rows_out_convert"] + dropped
    assert report["dropped_out_of_year"] > 0
    assert report["files"] >= 4
    assert report["rows_per_file"]["max_over_median"] < 2
    for f in fs.glob(f"{paths.data_dir()}/*.parquet"):
        h = pq.read_table(f, columns=["stac_hash"])["stac_hash"].to_numpy()
        assert np.all(np.diff(h.astype(np.int64)) >= 0)
    sql = f"SELECT count(*) FROM read_parquet('{paths.data_dir()}/*.parquet')"
    assert duckdb.sql(sql).fetchall()[0][0] == report["rows_out_compact"]


def test_dataset_manifest_lists_every_file(pipeline_run):
    fs, paths, report = pipeline_run
    manifest = pq.read_table(paths.dataset_manifest())
    assert manifest.num_rows == report["files"]
    assert sum(manifest["rows"].to_pylist()) == report["rows_out_compact"]


def test_finalize_detects_row_mismatch(pipeline_run):
    fs, paths, _ = pipeline_run
    fs.rm(fs.glob(f"{paths.data_dir()}/*.parquet")[0])
    fs.rm(fs.glob(f"{paths.compact_stats_dir()}/*.json")[0])
    with pytest.raises(ValueError, match="row count mismatch"):
        finalize_run(fs, paths)


class CountingFS(fsspec.implementations.local.LocalFileSystem):
    """Counts top-level reads; on s3fs one ``cat(list)`` is fetched concurrently."""

    cachable = False  # fsspec would otherwise hand every test the same counting instance

    def __init__(self):
        super().__init__()
        self.cats, self.opens, self._depth = 0, 0, 0

    def cat(self, path, *args, **kwargs):
        # Count only calls made by the code under test, not fsspec's internal ones.
        self.cats += self._depth == 0
        self._depth += 1
        try:
            return super().cat(path, *args, **kwargs)
        finally:
            self._depth -= 1

    def open(self, path, mode="rb", *args, **kwargs):
        self.opens += "r" in mode and self._depth == 0
        return super().open(path, mode, *args, **kwargs)


def test_finalize_batches_stats_reads(pipeline_run):
    _, paths, _ = pipeline_run
    fs = CountingFS()
    finalize_run(fs, paths)
    assert fs.cats == 2 and fs.opens == 0  # one batched read each for convert and compact stats


def test_compact_batches_fragment_reads(pipeline_run):
    _, paths, _ = pipeline_run
    fs = CountingFS()
    compact_range(fs, paths, 0, row_group_size=1000)
    assert fs.cats == 1 and fs.opens == 0
