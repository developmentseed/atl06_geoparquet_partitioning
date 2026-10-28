from pathlib import Path

from atl06_benchmark.config import Query, load_suite
from atl06_benchmark.sql import build_sql

SUITE = Path(__file__).resolve().parents[2] / "benchmark" / "queries.toml"


def test_build_sql_uses_only_plain_predicates():
    sql = build_sql("x/*.parquet", Query("q", (-1, -2, 3, 4), "2023-01-01", "2023-02-01", ["h_li"]))
    assert "stac_hash" not in sql
    assert "bbox.xmin >= -1" in sql and "bbox.xmax <= 3" in sql
    assert "bbox.ymin >= -2" in sql and "bbox.ymax <= 4" in sql
    assert "time_ns >= TIMESTAMP '2023-01-01'" in sql
    assert "time_ns < TIMESTAMP '2023-02-01'" in sql
    assert sql.startswith("SELECT h_li FROM read_parquet('x/*.parquet')")


def test_build_sql_without_time_or_bbox():
    assert "WHERE" not in build_sql("x.parquet", Query("q", None, None, None, ["h_li"]))
    assert "bbox" not in build_sql("x.parquet", Query("q", None, "2023-01-01", "2023-01-02", ["*"]))


def test_load_suite_default():
    suite = load_suite(SUITE)
    names = {q.name for q in suite.queries}
    assert names >= {"basin_full_year", "ice_sheet_one_day_greenland",
                     "ice_sheet_one_week_antarctica", "basin_one_week"}
    assert suite.repeats == 3
    jak = next(q for q in suite.queries if q.name == "basin_full_year")
    assert jak.bbox == (-50.5, 68.8, -48.5, 69.4) and jak.columns == ["time_ns", "h_li", "geometry"]
