from datetime import UTC, datetime
from types import SimpleNamespace

from atl06_processor import inventory
from atl06_processor.config import Settings
from atl06_processor.credentials import NsidcS3
from atl06_processor.inventory import build_inventory

NAMES = [f"ATL06_2023010{i}000000_0{i}000201_006_01.h5" for i in (3, 1, 2, 5, 4)]
GRANULES = [(n, f"s3://nsidc-cumulus-prod-protected/ATLAS/ATL06/006/{n}") for n in NAMES]


def test_build_inventory_assigns_srcid_sorted():
    table, items, sampled = build_inventory(GRANULES, sample_fraction=0.4, seed=0, limit=None)
    assert table["granule"].to_pylist() == sorted(NAMES)
    assert table["srcid"].to_pylist() == [0, 1, 2, 3, 4]
    assert [i["srcid"] for i in items] == [0, 1, 2, 3, 4]
    assert len(sampled) == 2 and all(s in items for s in sampled)


def test_build_inventory_limit_and_full_sample():
    table, items, sampled = build_inventory(GRANULES, sample_fraction=1.0, seed=0, limit=3)
    assert len(items) == 3 and sampled == items
    assert table.num_rows == 3


def test_build_inventory_samples_at_least_one():
    _, _, sampled = build_inventory(GRANULES, sample_fraction=0.0001, seed=0, limit=None)
    assert len(sampled) == 1


def test_search_granules_uses_direct_s3_links(monkeypatch):
    calls = {}

    def fake_search(**kwargs):
        calls.update(kwargs)
        return [SimpleNamespace(data_links=lambda access: [GRANULES[0][1]])]

    monkeypatch.setattr(inventory.earthaccess, "search_data", fake_search)
    start, end = datetime(2023, 1, 1, tzinfo=UTC), datetime(2024, 1, 1, tzinfo=UTC)
    assert inventory.search_granules(Settings(), start, end) == [GRANULES[0]]
    assert calls["short_name"] == "ATL06" and calls["version"] == "006"


def test_nsidc_creds_refresh_after_50_min():
    now = [0.0]
    fetched = []

    def fetch():
        fetched.append(now[0])
        return {"accessKeyId": "a", "secretAccessKey": "b", "sessionToken": str(now[0])}

    s3 = NsidcS3("secret", clock=lambda: now[0], fetch_credentials=fetch)
    s3.fs()
    now[0] = 49 * 60
    s3.fs()
    assert len(fetched) == 1
    now[0] = 51 * 60
    s3.fs()
    assert len(fetched) == 2
