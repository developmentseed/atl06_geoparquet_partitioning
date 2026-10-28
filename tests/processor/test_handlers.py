import json
from datetime import UTC, datetime, timedelta

import fsspec
import numpy as np
import pytest
from atl06_processor import handlers

from tests.conftest import BeamSpec, delta_time_for


@pytest.fixture
def local_env(monkeypatch, make_granule, tmp_path):
    fs = fsspec.filesystem("file")
    (tmp_path / "src").mkdir()
    rng = np.random.default_rng(0)
    granules = []
    for i in range(6):
        start = datetime(2023, 2, 1, tzinfo=UTC) + timedelta(days=40 * i)
        name = f"ATL06_{start:%Y%m%d%H%M%S}_{i:04d}0201_006_01.h5"
        n = 500
        dt = delta_time_for(start) + np.arange(n) * 0.05
        spec = BeamSpec(dt, rng.uniform(-88, 88, n), rng.uniform(-180, 180, n))
        granules.append((name, str(make_granule(tmp_path / "src", name, {"gt1l": spec}))))
    monkeypatch.setattr(handlers, "get_output_fs", lambda: fs)
    monkeypatch.setattr(handlers, "get_source_fs", lambda: fs)
    monkeypatch.setattr(handlers, "search_granules", lambda settings, start, end: granules)
    monkeypatch.setenv("ATL06_BUCKET", str(tmp_path / "bucket"))
    monkeypatch.setenv("ATL06_SAMPLE_GRANULE_FRACTION", "0.5")
    monkeypatch.setenv("ATL06_SAMPLE_ROW_FRACTION", "1.0")
    monkeypatch.setenv("ATL06_TARGET_ROWS_PER_FILE", "750")
    return fs, tmp_path / "bucket"


def test_handlers_end_to_end_local(local_env):
    fs, bucket = local_env
    batch_input = {"run_prefix": "runs/t"}
    inv = handlers.inventory_handler({"run_prefix": "runs/t"})
    assert inv["granules"] == 6 and inv["sampled"] == 3
    assert inv["bucket"] == str(bucket) and inv["sample_key"] == "runs/t/manifest/sample.json"

    sampled = json.loads(fs.cat(f"{bucket}/{inv['sample_key']}"))
    handlers.sample_handler({"Items": sampled, "BatchInput": batch_input})
    plan = handlers.plan_splits_handler({"run_prefix": "runs/t"})
    assert plan["ranges"] == 4  # est 3000 rows / 750 per file

    items = json.loads(fs.cat(f"{bucket}/{inv['granules_key']}"))
    out = handlers.convert_handler({"Items": items, "BatchInput": batch_input})
    assert sum(o["rows_out"] for o in out) == 3000

    ranges = json.loads(fs.cat(f"{bucket}/{plan['ranges_key']}"))
    handlers.compact_handler({"Items": ranges, "BatchInput": batch_input})
    report = handlers.finalize_handler({"run_prefix": "runs/t"})
    assert report["rows_out_compact"] == 3000 and report["files"] >= 2
