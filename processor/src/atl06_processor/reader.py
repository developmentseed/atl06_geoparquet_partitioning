"""Stream an ATL06 HDF5 granule into Arrow record batches, one per beam."""

from collections.abc import Iterator

import h5py
import numpy as np
import pyarrow as pa

from atl06_processor.geometry import bbox_struct, points_to_wkb
from atl06_processor.granule import GranuleInfo
from atl06_processor.schema import GT_CODES, PRE_HASH_SCHEMA

GPS_EPOCH_UNIX_S = 315_964_800  # 1980-01-06T00:00:00Z
# GPS - UTC offset; constant at 18 s for all dates since 2017-01-01 (covers the ICESat-2 era).
GPS_UTC_LEAP_S = 18

# output column -> dataset path under gtXX/land_ice_segments/
_VARIABLES: dict[str, str] = {
    "tide_ocean": "geophysical/tide_ocean",
    "n_fit_photons": "fit_statistics/n_fit_photons",
    "w_surface_window_final": "fit_statistics/w_surface_window_final",
    "atl06_quality_summary": "atl06_quality_summary",
    "r_eff": "geophysical/r_eff",
    "seg_azimuth": "ground_track/seg_azimuth",
    "bsnow_h": "geophysical/bsnow_h",
    "h_robust_sprd": "fit_statistics/h_robust_sprd",
    "sigma_geo_h": "sigma_geo_h",
    "h_li_sigma": "h_li_sigma",
    "dh_fit_dx": "fit_statistics/dh_fit_dx",
    "x_atc": "ground_track/x_atc",
    "y_atc": "ground_track/y_atc",
    "bsnow_conf": "geophysical/bsnow_conf",
    "h_li": "h_li",
    "segment_id": "segment_id",
}


def _read(ds: h5py.Dataset, type_: pa.DataType) -> pa.Array:
    values = ds[()]
    fill = ds.attrs.get("_FillValue")
    mask = None if fill is None else values == np.asarray(fill).reshape(-1)[0]
    return pa.array(values, mask=mask).cast(type_)


def _time_ns(delta_time: np.ndarray, gps_epoch: float) -> pa.Array:
    gps_ns = np.round((gps_epoch + delta_time) * 1e9).astype(np.int64)
    unix_ns = gps_ns + (GPS_EPOCH_UNIX_S - GPS_UTC_LEAP_S) * 1_000_000_000
    return pa.array(unix_ns, pa.timestamp("ns"))


def _const(value: int, n: int, type_: pa.DataType) -> pa.Array:
    return pa.array(np.full(n, value), type_)


def iter_beam_batches(h5: h5py.File, info: GranuleInfo, srcid: int) -> Iterator[pa.RecordBatch]:
    gps_epoch = float(h5["ancillary_data/atlas_sdp_gps_epoch"][0])
    for gt, gt_code in GT_CODES.items():
        lis = h5.get(f"{gt}/land_ice_segments")
        if lis is None or lis["delta_time"].shape[0] == 0:
            continue
        n = lis["delta_time"].shape[0]
        spot = int(np.asarray(h5[gt].attrs["atlas_spot_number"]).item())
        lon = lis["longitude"][()]
        lat = lis["latitude"][()]
        columns: dict[str, pa.Array] = {
            "time_ns": _time_ns(lis["delta_time"][()], gps_epoch),
            "region": _const(info.region, n, pa.uint8()),
            "gt": _const(gt_code, n, pa.uint8()),
            "rgt": _const(info.rgt, n, pa.uint16()),
            "spot": _const(spot, n, pa.uint8()),
            "srcid": _const(srcid, n, pa.int32()),
            "cycle": _const(info.cycle, n, pa.uint8()),
            "geometry": points_to_wkb(lon, lat),
            "bbox": bbox_struct(lon, lat),
        }
        for name, path in _VARIABLES.items():
            columns[name] = _read(lis[path], PRE_HASH_SCHEMA.field(name).type)
        yield pa.RecordBatch.from_arrays(
            [columns[f.name] for f in PRE_HASH_SCHEMA], schema=PRE_HASH_SCHEMA
        )
