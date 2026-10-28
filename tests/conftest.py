"""Synthetic ATL06-shaped HDF5 granules for tests."""

from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import h5py
import numpy as np
import pytest

ATLAS_SDP_GPS_EPOCH = 1198800018.0  # 2018-01-01T00:00:00 UTC in GPS seconds
FLOAT_FILL = np.float32(3.4028235e38)
GPS_EPOCH = datetime(1980, 1, 6, tzinfo=UTC)
GPS_UTC_LEAP_S = 18


def delta_time_for(utc: datetime) -> float:
    """ATL06 delta_time that corresponds to a UTC instant."""
    gps_seconds = (utc - GPS_EPOCH).total_seconds() + GPS_UTC_LEAP_S
    return gps_seconds - ATLAS_SDP_GPS_EPOCH


@dataclass
class BeamSpec:
    delta_time: np.ndarray
    lat: np.ndarray
    lon: np.ndarray
    spot: int = 1
    overrides: dict[str, np.ndarray] = field(default_factory=dict)


_FLOAT_VARS = {
    "h_li": "", "h_li_sigma": "", "sigma_geo_h": "",
    "n_fit_photons": "fit_statistics", "w_surface_window_final": "fit_statistics",
    "h_robust_sprd": "fit_statistics", "dh_fit_dx": "fit_statistics",
    "y_atc": "ground_track", "seg_azimuth": "ground_track",
    "tide_ocean": "geophysical", "bsnow_h": "geophysical", "r_eff": "geophysical",
}


def write_granule(directory: Path, name: str, beams: dict[str, BeamSpec]) -> Path:
    path = Path(directory) / name
    with h5py.File(path, "w") as h5:
        h5["ancillary_data/atlas_sdp_gps_epoch"] = np.array([ATLAS_SDP_GPS_EPOCH])
        for gt, spec in beams.items():
            n = len(spec.delta_time)
            grp = h5.create_group(gt)
            grp.attrs["atlas_spot_number"] = str(spec.spot).encode()
            lis = grp.create_group("land_ice_segments")

            def put(sub: str, var: str, values: np.ndarray, fill=None, lis=lis):
                g = lis.require_group(sub) if sub else lis
                ds = g.create_dataset(var, data=values)
                if fill is not None:
                    ds.attrs["_FillValue"] = fill

            put("", "delta_time", np.asarray(spec.delta_time, np.float64))
            put("", "latitude", np.asarray(spec.lat, np.float64))
            put("", "longitude", np.asarray(spec.lon, np.float64))
            put("", "segment_id", np.arange(n, dtype=np.int32) + 1000)
            put("", "atl06_quality_summary", np.zeros(n, np.int8))
            put("ground_track", "x_atc", np.arange(n, dtype=np.float64) * 20.0)
            put("geophysical", "bsnow_conf", spec.overrides.get("bsnow_conf", np.zeros(n, np.int8)),
                np.int8(127))
            for var, sub in _FLOAT_VARS.items():
                if var == "n_fit_photons":
                    put(sub, var, spec.overrides.get(var, np.full(n, 50, np.int32)),
                        np.int32(2147483647))
                    continue
                default = np.linspace(1.0, 2.0, n, dtype=np.float32)
                put(sub, var, np.asarray(spec.overrides.get(var, default), np.float32), FLOAT_FILL)
    return path


@pytest.fixture
def make_granule():
    return write_granule
