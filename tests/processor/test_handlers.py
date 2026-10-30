from datetime import UTC, datetime, timedelta

import numpy as np
import obstore
import pytest
from atl06_processor import handlers, storage
from obstore.store import LocalStore

from tests.conftest import BeamSpec, delta_time_for
from tests.processor.test_end_to_end import read_calls  # noqa: F401  (fixture)


@pytest.fixture
def local_env(monkeypatch, make_granule, tmp_path):
    (tmp_path / "src").mkdir()
    rng = np.random.default_rng(0)
    granules = []
    for i in range(6):
        start = datetime(2023, 2, 1, tzinfo=UTC) + timedelta(days=40 * i)
        name = f"ATL06_{start:%Y%m%d%H%M%S}_{i:04d}0201_006_01.h5"
        n = 500
        dt = delta_time_for(start) + np.arange(n) * 0.05
        spec = BeamSpec(dt, rng.uniform(-88, 88, n), rng.uniform(-180, 180, n))
        make_granule(tmp_path / "src", name, {"gt1l": spec})
        granules.append((name, f"s3://nsidc-test/ATLAS/ATL06/006/{name}"))
    src = LocalStore(tmp_path / "src")
    out = LocalStore(tmp_path / "bucket", mkdir=True)
    monkeypatch.setattr(handlers, "get_output_store", lambda: out)
    monkeypatch.setattr(handlers, "get_output_root", lambda: (tmp_path / "bucket").as_uri())
    monkeypatch.setattr(
        handlers, "get_source", lambda url: (src, storage.split_s3_url(url)[1].rsplit("/", 1)[1])
    )
    monkeypatch.setattr(handlers, "search_granules", lambda settings, start, end: granules)
    monkeypatch.setenv("ATL06_BUCKET", "test-bucket")
    monkeypatch.setenv("ATL06_SAMPLE_GRANULE_FRACTION", "0.5")
    monkeypatch.setenv("ATL06_SAMPLE_ROW_FRACTION", "1.0")
    monkeypatch.setenv("ATL06_TARGET_ROWS_PER_FILE", "750")
    return out


def test_handlers_end_to_end_local(local_env):
    store = local_env
    batch_input = {"run_prefix": "runs/t"}
    inv = handlers.inventory_handler({"run_prefix": "runs/t"})
    assert inv["granules"] == 6 and inv["sampled"] == 3
    assert inv["bucket"] == "test-bucket" and inv["sample_key"] == "runs/t/manifest/sample.json"

    sampled = storage.get_json(store, inv["sample_key"])
    handlers.sample_handler({"Items": sampled, "BatchInput": batch_input})
    plan = handlers.plan_splits_handler({"run_prefix": "runs/t"})
    assert plan["ranges"] == 4  # est 3000 rows / 750 per file

    items = storage.get_json(store, inv["granules_key"])
    out = handlers.convert_handler({"Items": items, "BatchInput": batch_input})
    assert sum(o["rows_out"] for o in out) == 3000

    ranges = storage.get_json(store, plan["ranges_key"])
    handlers.compact_handler({"Items": ranges, "BatchInput": batch_input})
    report = handlers.finalize_handler({"run_prefix": "runs/t"})
    assert report["rows_out_compact"] == 3000 and report["files"] >= 2
    assert report["iceberg_metadata"].endswith(".metadata.json")
    assert "/runs/t/iceberg/metadata/" in report["iceberg_metadata"]


def full_run(store, prefix="runs/t"):
    batch_input = {"run_prefix": prefix}
    inv = handlers.inventory_handler({"run_prefix": prefix})
    handlers.sample_handler({"Items": storage.get_json(store, inv["sample_key"]),
                             "BatchInput": batch_input})
    plan = handlers.plan_splits_handler({"run_prefix": prefix})
    handlers.convert_handler({"Items": storage.get_json(store, inv["granules_key"]),
                              "BatchInput": batch_input})
    handlers.compact_handler({"Items": storage.get_json(store, plan["ranges_key"]),
                              "BatchInput": batch_input})
    return plan, handlers.finalize_handler({"run_prefix": prefix})


def test_prepare_compact_requires_splits(local_env):
    with pytest.raises(ValueError, match="splits"):
        handlers.prepare_compact_handler({"run_prefix": "runs/never-planned"})


def test_prepare_compact_clears_previous_outputs(local_env):
    store = local_env
    plan, first = full_run(store)
    stale = ["runs/t/data/part-99999-00.parquet", "runs/t/stats/compact/99999-00.json",
             "runs/t/iceberg/metadata/stale.metadata.json"]
    for key in stale:
        storage.put_bytes(store, key, b"x")
    prepared = handlers.prepare_compact_handler({"run_prefix": "runs/t", "from_step": "compact"})
    assert prepared == {"ranges_key": plan["ranges_key"], "ranges": plan["ranges"]}
    for prefix in ("runs/t/data", "runs/t/stats/compact", "runs/t/iceberg"):
        assert storage.list_keys(store, prefix) == []
    assert storage.list_keys(store, "runs/t/staging")  # Convert output is kept
    handlers.compact_handler({"Items": storage.get_json(store, plan["ranges_key"]),
                              "BatchInput": {"run_prefix": "runs/t"}})
    second = handlers.finalize_handler({"run_prefix": "runs/t"})
    assert second["rows_out_compact"] == first["rows_out_compact"] == 3000
    assert second["files"] == first["files"]


def test_inventory_refuses_existing_run_prefix(local_env):
    handlers.inventory_handler({"run_prefix": "runs/t"})
    with pytest.raises(ValueError, match="already exists"):
        handlers.inventory_handler({"run_prefix": "runs/t"})


def test_inventory_overwrite_clears_previous_run(local_env):
    store = local_env
    handlers.inventory_handler({"run_prefix": "runs/t"})
    stale = "runs/t/staging/range=00000/old.parquet"
    storage.put_bytes(store, stale, b"x")
    handlers.inventory_handler({"run_prefix": "runs/t", "overwrite": True})
    assert stale not in storage.list_keys(store, "runs/t")
    assert "runs/t/manifest/granules.json" in storage.list_keys(store, "runs/t")


def test_plan_splits_batches_sample_reads(local_env, read_calls):  # noqa: F811
    store = local_env
    inv = handlers.inventory_handler({"run_prefix": "runs/t"})
    sampled = storage.get_json(store, inv["sample_key"])
    handlers.sample_handler({"Items": sampled, "BatchInput": {"run_prefix": "runs/t"}})
    read_calls.update(get_many=0, read_bytes=0)
    handlers.plan_splits_handler({"run_prefix": "runs/t"})
    # granules.json + sample.json, then one concurrent batch for every sample part
    assert read_calls == {"get_many": 1, "read_bytes": 2, "download_many": 0}


def test_inventory_rejects_window_outside_configured_year(local_env):
    with pytest.raises(ValueError, match="outside ATL06_YEAR"):
        handlers.inventory_handler({"run_prefix": "runs/w", "start": "2022-07-01T00:00:00+00:00"})


def test_source_store_is_cached_per_bucket(monkeypatch):
    made = []
    monkeypatch.setattr(handlers, "_nsidc_store", lambda bucket: made.append(bucket) or bucket)
    handlers._source_store.cache_clear()
    assert handlers.get_source("s3://b1/x/a.h5") == ("b1", "x/a.h5")
    assert handlers.get_source("s3://b1/y/b.h5") == ("b1", "y/b.h5")
    assert handlers.get_source("s3://b2/z.h5") == ("b2", "z.h5")
    assert made == ["b1", "b2"]


def test_inventory_fails_loudly_when_search_finds_nothing(local_env, monkeypatch):
    monkeypatch.setattr(handlers, "search_granules", lambda settings, start, end: [])
    with pytest.raises(ValueError, match=r"no ATL06 v007 granules"):
        handlers.inventory_handler({"run_prefix": "runs/empty"})
    assert not storage.prefix_exists(local_env, "runs/empty")  # nothing half-written


def test_prepare_compact_requires_complete_convert(local_env):
    store = local_env
    full_run(store)
    obstore.delete(store, storage.list_keys(store, "runs/t/stats/convert")[0])
    with pytest.raises(ValueError, match="Convert finished 5 of 6 granules"):
        handlers.prepare_compact_handler({"run_prefix": "runs/t", "from_step": "compact"})
    assert storage.list_keys(store, "runs/t/data")  # nothing deleted


def test_prepare_compact_validates_before_deleting(local_env):
    store = local_env
    plan, _ = full_run(store)
    obstore.delete(store, plan["ranges_key"])
    with pytest.raises(ValueError, match="ranges.json"):
        handlers.prepare_compact_handler({"run_prefix": "runs/t", "from_step": "compact"})
    assert storage.list_keys(store, "runs/t/data") and storage.list_keys(store, "runs/t/iceberg")
