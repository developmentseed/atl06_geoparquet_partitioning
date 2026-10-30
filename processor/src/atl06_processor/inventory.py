"""Granule discovery (CMR) and the run's granule table."""

import math
from datetime import UTC, datetime

import numpy as np
import pyarrow as pa
import requests

from atl06_processor.config import Settings

CMR_GRANULES = "https://cmr.earthdata.nasa.gov/search/granules.umm_json"
CMR_PAGE_SIZE = 2000  # CMR's maximum


def _session() -> requests.Session:
    return requests.Session()


def _cmr_time(t: datetime) -> str:
    return t.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _direct_h5_url(item: dict) -> str:
    for link in item["umm"].get("RelatedUrls", []):
        url = link.get("URL", "")
        if link.get("Type") == "GET DATA VIA DIRECT ACCESS" and url.endswith(".h5"):
            return url
    name = item.get("meta", {}).get("native-id", "?")
    raise ValueError(f"granule {name} has no direct-access s3:// .h5 link in CMR")


def search_granules(settings: Settings, start: datetime, end: datetime) -> list[tuple[str, str]]:
    """(granule_name, s3_url) for every granule in [start, end). Needs network, not EDL.

    Pages CMR directly and keeps only the direct-access link: each granule's full UMM record
    (~100 KB, mostly browse-image links) would otherwise be held for the whole year (~6 GB).
    """
    session = _session()
    params = {
        "short_name": settings.short_name,
        "version": settings.version,
        "temporal": f"{_cmr_time(start)},{_cmr_time(end)}",
        "page_size": CMR_PAGE_SIZE,
    }
    out, headers = [], {}
    while True:
        response = session.get(CMR_GRANULES, params=params, headers=headers, timeout=120)
        response.raise_for_status()
        items = response.json()["items"]
        if not items:
            return out
        for item in items:
            url = _direct_h5_url(item)
            out.append((url.rsplit("/", 1)[-1], url))
        headers = {"CMR-Search-After": response.headers["CMR-Search-After"]}


def build_inventory(
    granules: list[tuple[str, str]], sample_fraction: float, seed: int, limit: int | None
) -> tuple[pa.Table, list[dict], list[dict]]:
    """Assign srcid by sorted granule name; pick a random sample (at least one granule).

    ``limit`` keeps N granules evenly spaced through the sorted (time-ordered) list, so a
    small spike still spans the whole period.
    """
    ordered = sorted(granules)
    if limit is not None and limit < len(ordered):
        keep = np.unique(np.linspace(0, len(ordered) - 1, limit).round().astype(int))
        ordered = [ordered[i] for i in keep]
    items = [{"srcid": i, "granule": n, "s3_url": u} for i, (n, u) in enumerate(ordered)]
    table = pa.table(
        {
            "srcid": pa.array([i["srcid"] for i in items], pa.int32()),
            "granule": [i["granule"] for i in items],
            "s3_url": [i["s3_url"] for i in items],
        }
    )
    if sample_fraction >= 1:
        return table, items, items
    k = min(len(items), max(1, math.ceil(len(items) * sample_fraction)))
    picks = np.sort(np.random.default_rng(seed).choice(len(items), size=k, replace=False))
    return table, items, [items[i] for i in picks]
