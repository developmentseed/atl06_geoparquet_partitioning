import io

import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from atl06_processor import storage
from obstore.store import LocalStore, MemoryStore


@pytest.fixture
def store(tmp_path):
    return LocalStore(tmp_path, mkdir=True)


def test_put_and_read_parquet(store):
    table = pa.table({"a": [1, 2, 3]})
    storage.put_parquet(store, "x/y/t.parquet", table, compression="zstd")
    assert pq.read_table(pa.BufferReader(storage.read_bytes(store, "x/y/t.parquet"))) == table
    assert storage.size(store, "x/y/t.parquet") > 0


def test_put_and_get_json(store):
    storage.put_json(store, "m/a.json", [{"k": 1}])
    assert storage.get_json(store, "m/a.json") == [{"k": 1}]


def test_list_keys_filters_suffix_and_sorts(store):
    for k in ("p/b.json", "p/a.json", "p/c.parquet", "q/d.json"):
        storage.put_json(store, k, {})
    assert storage.list_keys(store, "p", ".json") == ["p/a.json", "p/b.json"]
    assert storage.list_keys(store, "missing", ".json") == []


def test_get_many_returns_every_key(store):
    keys = [f"f/{i}.json" for i in range(20)]
    for i, k in enumerate(keys):
        storage.put_json(store, k, i)
    got = storage.get_many(store, keys)
    assert sorted(got) == sorted(keys)
    assert bytes(got["f/7.json"]) == b"7"
    assert storage.get_many(store, []) == {}


def test_delete_prefix(store):
    for k in ("r/a.json", "r/s/b.json", "rr/c.json"):
        storage.put_json(store, k, 1)
    assert storage.prefix_exists(store, "r")
    storage.delete_prefix(store, "r")
    assert not storage.prefix_exists(store, "r")
    assert storage.prefix_exists(store, "rr")  # sibling with a shared name prefix is untouched


def test_read_bytes_opens_in_h5py():
    store = MemoryStore()
    buf = io.BytesIO()
    with h5py.File(buf, "w") as f:
        f["x"] = np.arange(5)
    storage.put_bytes(store, "g.h5", buf.getvalue())
    with h5py.File(io.BytesIO(storage.read_bytes(store, "g.h5")), "r") as f:
        assert list(f["x"][()]) == [0, 1, 2, 3, 4]


def test_split_s3_url():
    assert storage.split_s3_url("s3://bucket/a/b/c.h5") == ("bucket", "a/b/c.h5")
    with pytest.raises(ValueError):
        storage.split_s3_url("https://bucket/a.h5")
