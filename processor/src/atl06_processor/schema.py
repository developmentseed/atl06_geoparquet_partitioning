"""Output schema (SlideRule-style ATL06 columns) and GeoParquet metadata."""

import json

import pyarrow as pa
import pyarrow.compute as pc
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


def _output_fields() -> list[pa.Field]:
    fields = []
    for f in SCHEMA:
        if f.name == "time_ns":
            # Iceberg v2 has no nanosecond timestamp, so keep SlideRule's values as plain int64
            # and add a microsecond timestamp for filtering.
            fields += [pa.field("time_ns", pa.int64()), pa.field("time", pa.timestamp("us"))]
        elif f.name == "geometry":
            # Top-level point coordinates: Iceberg manifests carry bounds for these, not bbox.*.
            fields += [pa.field("longitude", pa.float64()), pa.field("latitude", pa.float64()), f]
        else:
            fields.append(f)
    return fields


# Schema of compacted output files; staging fragments keep SCHEMA.
OUTPUT_SCHEMA = pa.schema(_output_fields())


def to_output(data: pa.Table | pa.RecordBatch) -> pa.Table:
    """Converts rows in SCHEMA to OUTPUT_SCHEMA."""
    table = pa.Table.from_batches([data]) if isinstance(data, pa.RecordBatch) else data
    bbox = table["bbox"]
    derived = {
        "time_ns": pc.cast(table["time_ns"], pa.int64()),
        "time": pc.cast(table["time_ns"], pa.timestamp("us"), safe=False),
        "longitude": pc.struct_field(bbox, "xmin"),
        "latitude": pc.struct_field(bbox, "ymin"),
    }
    columns = [derived[f.name] if f.name in derived else table[f.name] for f in OUTPUT_SCHEMA]
    return pa.Table.from_arrays(columns, schema=OUTPUT_SCHEMA)


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


def nan_counts(table: pa.Table) -> dict[str, int]:
    """NaN count per floating-point column (nulls are not NaN)."""
    return {
        f.name: pc.sum(pc.is_nan(table[f.name])).as_py() or 0
        for f in table.schema
        if pa.types.is_floating(f.type)
    }
