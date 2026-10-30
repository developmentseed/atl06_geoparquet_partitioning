from datetime import UTC, datetime
from types import SimpleNamespace

import pytest
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


def umm_item(url_type_pairs):
    return {"umm": {"RelatedUrls": [{"Type": t, "URL": u} for t, u in url_type_pairs]}}


def granule_item(name):
    base = "nsidc-cumulus-prod-protected/ATLAS/ATL06/007/2023/01/01"
    return umm_item([
        ("GET DATA", f"https://data.nsidc.earthdatacloud.nasa.gov/{base}/{name}"),
        ("GET DATA VIA DIRECT ACCESS", f"s3://{base}/{name}"),
        ("VIEW RELATED INFORMATION", f"s3://{base}/{name.removesuffix('.h5')}.iso.xml"),
    ])


class FakeCMR:
    """Two pages linked by CMR-Search-After, then an empty page."""

    def __init__(self, pages):
        self.pages, self.requests = pages, []

    def get(self, url, params, headers, timeout):
        self.requests.append((url, dict(params), dict(headers)))
        i = len(self.requests) - 1
        items = self.pages[i] if i < len(self.pages) else []
        return SimpleNamespace(
            json=lambda: {"items": items},
            headers={"CMR-Search-After": f"token-{i}"} if items else {},
            raise_for_status=lambda: None,
        )


def test_search_granules_pages_cmr_and_keeps_direct_h5_links(monkeypatch):
    names = [f"ATL06_2023010{i}000000_0{i}000201_007_01.h5" for i in range(1, 4)]
    cmr = FakeCMR([[granule_item(names[0]), granule_item(names[1])], [granule_item(names[2])]])
    monkeypatch.setattr(inventory, "_session", lambda: cmr)
    start, end = datetime(2023, 1, 1, tzinfo=UTC), datetime(2024, 1, 1, tzinfo=UTC)
    out = inventory.search_granules(Settings(), start, end)
    base = "s3://nsidc-cumulus-prod-protected/ATLAS/ATL06/007/2023/01/01"
    assert out == [(n, f"{base}/{n}") for n in names]
    url, params, headers = cmr.requests[0]
    assert url.endswith("/search/granules.umm_json")
    assert params["short_name"] == "ATL06" and params["version"] == "007"  # v006 is retired
    assert params["temporal"] == "2023-01-01T00:00:00Z,2024-01-01T00:00:00Z"
    assert "CMR-Search-After" not in headers
    assert cmr.requests[1][2]["CMR-Search-After"] == "token-0"
    assert len(cmr.requests) == 3


def test_search_granules_names_a_granule_without_direct_link(monkeypatch):
    cmr = FakeCMR([[{"meta": {"native-id": "ATL06_X.h5"}, **umm_item([("GET DATA", "https://x/a.h5")])}]])
    monkeypatch.setattr(inventory, "_session", lambda: cmr)
    start, end = datetime(2023, 1, 1, tzinfo=UTC), datetime(2024, 1, 1, tzinfo=UTC)
    with pytest.raises(ValueError, match="ATL06_X.h5"):
        inventory.search_granules(Settings(), start, end)


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
