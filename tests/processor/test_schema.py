import pyarrow as pa
import pyarrow.compute as pc
from atl06_processor.schema import GT_CODES, OUTPUT_SCHEMA, SCHEMA, geo_metadata, to_output


def test_schema_column_order():
    assert SCHEMA.names == [
        "time_ns", "tide_ocean", "region", "gt", "n_fit_photons",
        "w_surface_window_final", "atl06_quality_summary", "rgt", "r_eff", "seg_azimuth",
        "bsnow_h", "h_robust_sprd", "sigma_geo_h", "h_li_sigma", "dh_fit_dx", "x_atc", "spot",
        "y_atc", "srcid", "bsnow_conf", "cycle", "h_li", "segment_id", "geometry", "stac_hash",
        "bbox",
    ]


def test_schema_types():
    assert SCHEMA.field("time_ns").type == pa.timestamp("ns")
    assert SCHEMA.field("x_atc").type == pa.float64()
    assert SCHEMA.field("h_li").type == pa.float32()
    assert SCHEMA.field("rgt").type == pa.uint16()
    assert SCHEMA.field("segment_id").type == pa.uint32()
    assert SCHEMA.field("stac_hash").type == pa.uint64()
    assert SCHEMA.field("bbox").type == pa.struct(
        [(k, pa.float64()) for k in ("xmin", "ymin", "xmax", "ymax")]
    )
    assert SCHEMA.field("geometry").metadata[b"ARROW:extension:name"] == b"geoarrow.wkb"


def test_gt_codes():
    assert GT_CODES == {"gt1l": 10, "gt1r": 20, "gt2l": 30, "gt2r": 40, "gt3l": 50, "gt3r": 60}


def test_geo_metadata_has_covering():
    md = geo_metadata((-1.0, -2.0, 3.0, 4.0))
    col = md["columns"]["geometry"]
    assert md["version"] == "1.1.0" and col["encoding"] == "WKB"
    assert col["covering"]["bbox"]["xmin"] == ["bbox", "xmin"]
    assert col["bbox"] == [-1.0, -2.0, 3.0, 4.0]


def test_output_schema_layout():
    names = OUTPUT_SCHEMA.names
    assert names[:2] == ["time_ns", "time"]
    assert names[names.index("geometry") - 2 : names.index("geometry")] == ["longitude", "latitude"]
    assert set(names) == set(SCHEMA.names) | {"time", "longitude", "latitude"}
    assert OUTPUT_SCHEMA.field("time_ns").type == pa.int64()
    assert OUTPUT_SCHEMA.field("time").type == pa.timestamp("us")
    assert OUTPUT_SCHEMA.field("longitude").type == pa.float64()
    assert OUTPUT_SCHEMA.field("latitude").type == pa.float64()
    # Iceberg v2 (via pyiceberg) rejects any nanosecond timestamp column in a file.
    assert not any(pa.types.is_timestamp(f.type) and f.type.unit == "ns" for f in OUTPUT_SCHEMA)
    assert OUTPUT_SCHEMA.field("geometry").metadata == SCHEMA.field("geometry").metadata


NS = [1_688_169_600_123_456_789, None, 1_700_000_000_000_000_001]


def _source_table():
    n = 3
    cols = {f.name: pa.nulls(n, f.type) for f in SCHEMA}
    cols["time_ns"] = pa.array(NS, pa.timestamp("ns"))
    cols["bbox"] = pa.StructArray.from_arrays(
        [pa.array([1.5, 2.5, -3.5]), pa.array([-70.0, 80.0, 0.25]),
         pa.array([1.5, 2.5, -3.5]), pa.array([-70.0, 80.0, 0.25])],
        names=["xmin", "ymin", "xmax", "ymax"],
    )
    return pa.table(cols, schema=SCHEMA)


def test_to_output_values():
    src = _source_table()
    out = to_output(src)
    assert out.schema.equals(OUTPUT_SCHEMA, check_metadata=False)
    assert out["time_ns"].to_pylist() == NS
    assert out["time"].to_pylist() == pc.cast(
        src["time_ns"], pa.timestamp("us"), safe=False).to_pylist()
    assert out["longitude"].to_pylist() == [1.5, 2.5, -3.5]
    assert out["latitude"].to_pylist() == [-70.0, 80.0, 0.25]
    assert out["bbox"].equals(src["bbox"])


def test_to_output_accepts_record_batch():
    batch = _source_table().to_batches()[0]
    assert to_output(batch).num_rows == 3
