import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from atl06_benchmark.config import Dataset, Query, Suite
from atl06_benchmark.runner import run_query, run_suite
from atl06_benchmark.sql import build_sql


def write_sorted_file(path, n=10_000, row_group_size=1_000, xrange=(-180, 180)):
    x = np.linspace(*xrange, n)
    y = np.linspace(-90, 90, n)
    bbox = pa.StructArray.from_arrays([pa.array(x), pa.array(y), pa.array(x), pa.array(y)],
                                      ["xmin", "ymin", "xmax", "ymax"])
    t = pa.array(np.arange(n) * 1_000_000_000 + 1_672_531_200_000_000_000, pa.timestamp("ns"))
    pq.write_table(pa.table({"time_ns": t, "h_li": np.ones(n, np.float32), "bbox": bbox}),
                   path, row_group_size=row_group_size)


def test_run_query_reports_pruning(tmp_path):
    write_sorted_file(tmp_path / "a.parquet")
    q = Query("narrow", (0.0, -90.0, 1.0, 90.0), None, None, ["h_li"])
    m = run_query(duckdb.connect(), build_sql(str(tmp_path / "a.parquet"), q), files_total=1)
    assert m.row_groups_total == 10 and m.row_groups_scanned <= 2
    assert m.footers_read == 1 and m.bytes_read > 0 and m.rows > 0 and m.wall_s > 0


def test_run_suite_one_row_per_repeat(tmp_path):
    write_sorted_file(tmp_path / "a.parquet")
    write_sorted_file(tmp_path / "b.parquet")
    suite = Suite(
        datasets=[Dataset("local", str(tmp_path / "*.parquet"))],
        queries=[Query("narrow", (0.0, -90.0, 1.0, 90.0), None, None, ["h_li"])],
        repeats=2,
    )
    table = run_suite(suite)
    assert table.num_rows == 2
    assert table["files_total"].to_pylist() == [2, 2]
    assert set(table["dataset"].to_pylist()) == {"local"}


def test_files_scanned_counts_only_files_with_matching_row_groups(tmp_path):
    for i, xr in enumerate([(-180, -100), (-10, 10), (100, 180)]):
        write_sorted_file(tmp_path / f"{i}.parquet", xrange=xr)
    suite = Suite(
        datasets=[Dataset("local", str(tmp_path / "*.parquet"))],
        queries=[Query("narrow", (0.0, -90.0, 1.0, 90.0), "2023-01-01", "2023-01-02", ["h_li"])],
        repeats=1,
    )
    row = run_suite(suite).to_pylist()[0]
    assert row["files_total"] == 3 and row["footers_read"] == 3
    assert row["files_scanned"] == 1
    assert row["row_groups_scanned"] <= 2 and row["row_groups_total"] == 30
