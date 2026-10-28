import pyarrow as pa
from atl06_processor.schema import GT_CODES, SCHEMA, geo_metadata


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
