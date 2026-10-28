"""Output schema (SlideRule-style ATL06 columns) and GeoParquet metadata."""

import json

import pyarrow as pa
from pyproj import CRS

GT_CODES: dict[str, int] = {
    "gt1l": 10, "gt1r": 20, "gt2l": 30, "gt2r": 40, "gt3l": 50, "gt3r": 60,
}

CRS84_PROJJSON: dict = CRS("OGC:CRS84").to_json_dict()

BBOX_TYPE = pa.struct([(k, pa.float64()) for k in ("xmin", "ymin", "xmax", "ymax")])

GEOMETRY_FIELD = pa.field(
    "geometry",
    pa.binary(),
    metadata={
        "ARROW:extension:name": "geoarrow.wkb",
        "ARROW:extension:metadata": json.dumps({"crs": CRS84_PROJJSON}),
    },
)

_COLUMNS: list[pa.Field] = [
    pa.field("time_ns", pa.timestamp("ns")),
    pa.field("tide_ocean", pa.float32()),
    pa.field("region", pa.uint8()),
    pa.field("gt", pa.uint8()),
    pa.field("n_fit_photons", pa.int32()),
    pa.field("w_surface_window_final", pa.float32()),
    pa.field("atl06_quality_summary", pa.int8()),
    pa.field("rgt", pa.uint16()),
    pa.field("r_eff", pa.float32()),
    pa.field("seg_azimuth", pa.float32()),
    pa.field("bsnow_h", pa.float32()),
    pa.field("h_robust_sprd", pa.float32()),
    pa.field("sigma_geo_h", pa.float32()),
    pa.field("h_li_sigma", pa.float32()),
    pa.field("dh_fit_dx", pa.float32()),
    pa.field("x_atc", pa.float64()),
    pa.field("spot", pa.uint8()),
    pa.field("y_atc", pa.float32()),
    pa.field("srcid", pa.int32()),
    pa.field("bsnow_conf", pa.int8()),
    pa.field("cycle", pa.uint8()),
    pa.field("h_li", pa.float32()),
    pa.field("segment_id", pa.uint32()),
    GEOMETRY_FIELD,
]

# Schema produced by the reader, before hashing.
PRE_HASH_SCHEMA = pa.schema([*_COLUMNS, pa.field("bbox", BBOX_TYPE)])

SCHEMA = pa.schema([*_COLUMNS, pa.field("stac_hash", pa.uint64()), pa.field("bbox", BBOX_TYPE)])


def geo_metadata(bbox: tuple[float, float, float, float]) -> dict:
    """GeoParquet 1.1 ``geo`` metadata with a bbox covering column."""
    return {
        "version": "1.1.0",
        "primary_column": "geometry",
        "columns": {
            "geometry": {
                "encoding": "WKB",
                "geometry_types": ["Point"],
                "crs": CRS84_PROJJSON,
                "bbox": [float(v) for v in bbox],
                "covering": {
                    "bbox": {k: ["bbox", k] for k in ("xmin", "ymin", "xmax", "ymax")}
                },
            }
        },
    }
