from datetime import UTC, datetime
from types import SimpleNamespace

from atl06_processor import inventory
from atl06_processor.config import Settings
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


def test_nsidc_store_uses_earthdata_provider(monkeypatch):
    from atl06_processor import credentials
    from obstore.auth.earthdata import NasaEarthdataCredentialProvider
    from obstore.store import S3Store

    monkeypatch.setattr(credentials, "earthdata_auth", lambda name: ("user", "pass"))
    store = credentials.nsidc_store("nsidc-cumulus-prod-protected", "atl06/earthdata")
    assert isinstance(store, S3Store)
    assert store.config["bucket"] == "nsidc-cumulus-prod-protected"
    assert isinstance(store.credential_provider, NasaEarthdataCredentialProvider)


def test_limit_spreads_across_the_year():
    names = [f"ATL06_2023{m:02d}01000000_0{m:02d}00201_006_01.h5" for m in range(1, 13)]
    granules = [(n, f"s3://b/{n}") for n in names]
    _, items, _ = build_inventory(granules, sample_fraction=1.0, seed=0, limit=4)
    months = [int(i["granule"][10:12]) for i in items]
    assert months[0] == 1 and months[-1] == 12 and len(set(months)) == 4
    assert [i["srcid"] for i in items] == [0, 1, 2, 3]
