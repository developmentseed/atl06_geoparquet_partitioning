"""Granule discovery (CMR via earthaccess) and the run's granule table."""

import math
from datetime import datetime

import earthaccess
import numpy as np
import pyarrow as pa

from atl06_processor.config import Settings


def search_granules(settings: Settings, start: datetime, end: datetime) -> list[tuple[str, str]]:
    """(granule_name, s3_url) for every granule in [start, end). Needs network, not EDL."""
    results = earthaccess.search_data(
        short_name=settings.short_name,
        version=settings.version,
        temporal=(start.isoformat(), end.isoformat()),
    )
    out = []
    for granule in results:
        url = next(u for u in granule.data_links(access="direct") if u.endswith(".h5"))
        out.append((url.rsplit("/", 1)[-1], url))
    return out


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
