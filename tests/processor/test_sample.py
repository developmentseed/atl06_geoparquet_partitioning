from datetime import UTC, datetime

import fsspec
import numpy as np
from atl06_processor.granule import parse_granule_name
from atl06_processor.paths import RunPaths
from atl06_processor.sample import sample_granule

from tests.conftest import BeamSpec, delta_time_for

NAME = "ATL06_20230601000000_01790201_006_01.h5"


def test_sample_granule_counts_and_subsamples(make_granule, tmp_path):
    n = 1000
    dt = delta_time_for(datetime(2023, 6, 1, tzinfo=UTC)) + np.arange(n) * 0.01
    spec = BeamSpec(dt, np.linspace(60, 80, n), np.linspace(-50, -40, n))
    path = make_granule(tmp_path, NAME, {"gt1l": spec})
    count, hashes = sample_granule(
        fsspec.filesystem("file"), str(path), parse_granule_name(NAME), 2023, 0.1, seed=0
    )
    assert count == 1000
    assert 50 <= len(hashes) <= 150 and hashes.dtype == np.uint64


def test_run_paths_layout():
    p = RunPaths("bucket/runs/x")
    assert p.fragment(3, NAME) == "bucket/runs/x/staging/range=00003/" + NAME[:-3] + ".parquet"
    assert p.data_file(12) == "bucket/runs/x/data/part-00012.parquet"
    assert p.sample_part(NAME) == "bucket/runs/x/sample/" + NAME[:-3] + ".parquet"
