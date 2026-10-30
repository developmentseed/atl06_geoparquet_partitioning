import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from atl06_benchmark.config import Dataset, Query, Suite
from atl06_benchmark.runner import run_query, run_suite
from atl06_benchmark.sql import build_sql
from pyiceberg.catalog.sql import SqlCatalog


def sorted_table(n=10_000, xrange=(-180, 180), day=0):
    x = np.linspace(*xrange, n)
    y = np.linspace(-90, 90, n)
    t = pa.array(np.arange(n) * 1_000 + (1_672_531_200 + day * 86_400) * 1_000_000,
                 pa.timestamp("us"))
    return pa.table({"time": t, "h_li": np.ones(n, np.float32), "longitude": x, "latitude": y})


def write_sorted_file(path, n=10_000, row_group_size=1_000, xrange=(-180, 180)):
    pq.write_table(sorted_table(n, xrange), path, row_group_size=row_group_size)


def test_run_query_reports_pruning(tmp_path):
    write_sorted_file(tmp_path / "a.parquet")
    q = Query("narrow", (0.0, -90.0, 1.0, 90.0), None, None, ["h_li"])
    sql = build_sql(Dataset("a", str(tmp_path / "a.parquet")), q)
    m = run_query(duckdb.connect(), sql, files_total=1)
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


def iceberg_table(tmp_path, n_files=5):
    """One file per day, registered with plain add_files (time-only queries prune)."""
    files = []
    for i in range(n_files):
        path = tmp_path / f"d{i}.parquet"
        pq.write_table(sorted_table(1_000, (-180 + 10 * i, -171 + 10 * i), day=i), path)
        files.append(str(path))
    catalog = SqlCatalog("c", uri=f"sqlite:///{tmp_path}/c.db", warehouse=tmp_path.as_uri())
    catalog.create_namespace("n")
    table = catalog.create_table("n.t", schema=pq.read_schema(files[0]))
    table.add_files(files)
    return catalog.load_table("n.t").metadata_location


def test_run_suite_on_iceberg_dataset(tmp_path):
    metadata = iceberg_table(tmp_path)
    suite = Suite(
        datasets=[Dataset("ice", metadata, "iceberg")],
        queries=[Query("day2", None, "2023-01-03", "2023-01-04", ["h_li"])],
        repeats=1,
    )
    row = run_suite(suite).to_pylist()[0]
    assert row["files_total"] == 5 and row["footers_read"] == 1
    assert row["files_scanned"] == 1 and row["rows"] == 1_000
