"""ATL06 granule naming: ATL06_YYYYMMDDhhmmss_ttttccrr_vvv_ee.h5."""

import re
from dataclasses import dataclass

_NAME_RE = re.compile(r"ATL06_\d{14}_(?P<rgt>\d{4})(?P<cycle>\d{2})(?P<region>\d{2})_\d{3}_\d{2}")


@dataclass(frozen=True)
class GranuleInfo:
    name: str
    rgt: int
    cycle: int
    region: int

    @property
    def stem(self) -> str:
        return self.name.removesuffix(".h5")


def parse_granule_name(name: str) -> GranuleInfo:
    m = _NAME_RE.search(name)
    if m is None:
        raise ValueError(f"not an ATL06 granule name: {name!r}")
    return GranuleInfo(name, int(m["rgt"]), int(m["cycle"]), int(m["region"]))
