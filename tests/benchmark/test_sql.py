from pathlib import Path

from atl06_benchmark.config import Dataset, Query, load_suite
from atl06_benchmark.sql import build_files_scanned_sql, build_sql, data_files_sql, source_sql

SUITE = Path(__file__).resolve().parents[2] / "benchmark" / "queries.default.toml"
PARQUET = Dataset("p", "x/*.parquet")
ICEBERG = Dataset("i", "s3://b/runs/r/iceberg/metadata/00001-a.metadata.json", "iceberg")


def test_build_sql_uses_only_plain_predicates():
    sql = build_sql(PARQUET, Query("q", (-1, -2, 3, 4), "2023-01-01", "2023-02-01", ["h_li"]))
    assert "stac_hash" not in sql and "bbox" not in sql
    assert "longitude >= -1" in sql and "longitude <= 3" in sql
    assert "latitude >= -2" in sql and "latitude <= 4" in sql
    assert "time >= TIMESTAMP '2023-01-01'" in sql
    assert "time < TIMESTAMP '2023-02-01'" in sql
    assert sql.startswith("SELECT h_li FROM read_parquet('x/*.parquet')")


def test_build_sql_without_time_or_bbox():
    assert "WHERE" not in build_sql(PARQUET, Query("q", None, None, None, ["h_li"]))
    sql = build_sql(PARQUET, Query("q", None, "2023-01-01", "2023-01-02", ["*"]))
    assert "longitude" not in sql


def test_build_sql_iceberg_source():
    assert source_sql(ICEBERG) == f"iceberg_scan('{ICEBERG.uri}')"
    sql = build_sql(ICEBERG, Query("q", None, "2023-01-01", "2023-01-02", ["h_li"]))
    assert sql.startswith(f"SELECT h_li FROM iceberg_scan('{ICEBERG.uri}') WHERE")


def test_data_files_sql():
    assert data_files_sql(PARQUET) == "SELECT file FROM glob('x/*.parquet') ORDER BY 1"
    sql = data_files_sql(ICEBERG)
    assert f"iceberg_metadata('{ICEBERG.uri}')" in sql and "DELETED" in sql


def test_files_scanned_sql_lists_files_and_uses_top_level_columns():
    sql = build_files_scanned_sql(["a.parquet", "b.parquet"],
                                  Query("q", (-1, -2, 3, 4), "2023-01-01", None, ["*"]))
    assert "parquet_metadata(['a.parquet', 'b.parquet'])" in sql
    assert "col = 'longitude'" in sql and "col = 'time'" in sql and "bbox" not in sql


def test_load_suite_default():
    suite = load_suite(SUITE)
    names = {q.name for q in suite.queries}
    assert names >= {"basin_full_year", "ice_sheet_one_day_greenland",
                     "ice_sheet_one_week_antarctica", "basin_one_week", "global_one_day"}
    assert suite.repeats == 3
    jak = next(q for q in suite.queries if q.name == "basin_full_year")
    assert jak.bbox == (-50.5, 68.8, -48.5, 69.4) and jak.columns == ["time", "h_li", "geometry"]
    day = next(q for q in suite.queries if q.name == "global_one_day")
    assert day.bbox is None and (day.start, day.end) == ("2023-07-01", "2023-07-02")
    assert {d.format for d in suite.datasets} == {"parquet", "iceberg"}


def test_load_suite_format_defaults_to_parquet(tmp_path):
    p = tmp_path / "s.toml"
    p.write_text('[[datasets]]\nname = "d"\nuri = "x/*.parquet"\n\n'
                 '[[queries]]\nname = "q"\nstart = "2023-01-01"\n')
    assert load_suite(p).datasets[0].format == "parquet"
