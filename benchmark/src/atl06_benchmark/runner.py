"""Run queries in DuckDB and read pruning metrics from its JSON profiler."""

import json
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import duckdb
import pyarrow as pa

from atl06_benchmark.config import Suite
from atl06_benchmark.sql import build_files_scanned_sql, build_sql


@dataclass
class QueryMetrics:
    dataset: str
    query: str
    repeat: int
    files_total: int
    footers_read: int  # DuckDB opens every file's footer to prune; not a pruning measure
    files_scanned: int  # files with >= 1 row group whose statistics can match the query
    row_groups_total: int
    row_groups_scanned: int
    bytes_read: int
    rows: int
    wall_s: float


def _table_scans(node: dict):
    if node.get("type") == "TABLE_SCAN":
        yield node
    for child in node.get("children", []):
        yield from _table_scans(child)


def run_query(
    con: duckdb.DuckDBPyConnection, sql: str, files_total: int,
    dataset: str = "", query: str = "", repeat: int = 0, files_scanned: int = -1,
) -> QueryMetrics:
    with tempfile.TemporaryDirectory() as tmp:
        profile = Path(tmp) / "profile.json"
        con.execute("SET enable_external_file_cache = false")
        con.execute("PRAGMA enable_profiling = 'json'")
        con.execute(f"PRAGMA profiling_output = '{profile}'")
        con.execute("SET profiling_mode = 'detailed'")
        start = time.perf_counter()
        rows = con.sql(sql).to_arrow_table().num_rows
        wall = time.perf_counter() - start
        con.execute("PRAGMA disable_profiling")
        p = json.loads(profile.read_text())
    scans = [s for op in p["operator"] for s in _table_scans(op)]
    return QueryMetrics(
        dataset=dataset,
        query=query,
        repeat=repeat,
        files_total=files_total,
        footers_read=sum(int(s["extra_info"].get("Total Files Read", 0)) for s in scans),
        files_scanned=files_scanned,
        row_groups_total=p["query"]["total_row_groups_to_scan"],
        row_groups_scanned=p["query"]["total_row_groups_scanned"],
        bytes_read=p["io"]["total_bytes_read"],
        rows=rows,
        wall_s=wall,
    )


def connect(suite: Suite) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    if any(d.uri.startswith("s3://") for d in suite.datasets):
        con.execute("INSTALL httpfs; LOAD httpfs; INSTALL aws; LOAD aws")
        con.execute("CREATE SECRET (TYPE s3, PROVIDER credential_chain)")
    return con


def run_suite(suite: Suite) -> pa.Table:
    con = connect(suite)
    results = []
    for ds in suite.datasets:
        files_total = con.execute(f"SELECT count(*) FROM glob('{ds.uri}')").fetchall()[0][0]
        for q in suite.queries:
            sql = build_sql(ds.uri, q)
            scanned = con.execute(build_files_scanned_sql(ds.uri, q)).fetchall()[0][0]
            for r in range(suite.repeats):
                metrics = run_query(con, sql, files_total, ds.name, q.name, r, scanned)
                results.append(asdict(metrics))
    return pa.Table.from_pylist(results)
