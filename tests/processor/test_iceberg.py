import json

import duckdb
import numpy as np
import pyarrow as pa
import pytest
from atl06_processor import storage
from atl06_processor.iceberg import register_table
from atl06_processor.paths import RunPaths
from atl06_processor.schema import OUTPUT_SCHEMA, SCHEMA, nan_counts, to_output
from obstore.store import LocalStore
from pyiceberg.table import StaticTable

from tests.processor.test_end_to_end import pipeline_run  # noqa: F401  (fixture)

DAY_NS = 86_400 * 10**9
JAN1_NS = 1_672_531_200 * 10**9  # 2023-01-01T00:00:00Z


def output_file(i: int, n: int = 100) -> pa.Table:
    """Rows on day ``10 * i`` of 2023, longitudes in the 10° band starting at ``-180 + 10 * i``."""
    rng = np.random.default_rng(i)
    cols = {f.name: pa.nulls(n, f.type) for f in SCHEMA}
    cols["time_ns"] = pa.array(JAN1_NS + 10 * i * DAY_NS + rng.integers(0, DAY_NS, n)).cast(
        pa.timestamp("ns"))
    x, y = rng.uniform(-180 + 10 * i, -170 + 10 * i, n), rng.uniform(-80, 80, n)
    cols["bbox"] = pa.StructArray.from_arrays(
        [pa.array(x), pa.array(y), pa.array(x), pa.array(y)],
        names=["xmin", "ymin", "xmax", "ymax"])
    cols["stac_hash"] = pa.array(np.arange(n, dtype=np.uint64) + i * n)
    return to_output(pa.table(cols, schema=SCHEMA))


@pytest.fixture
def table_files(tmp_path):
    store = LocalStore(tmp_path, mkdir=True)
    paths = RunPaths("run")
    keys, nans = [], {}
    for i in range(10):
        key = paths.data_file(i, 0)
        t = output_file(i)
        storage.put_parquet(store, key, t)
        keys.append(key)
        nans[key] = nan_counts(t)
    return tmp_path.as_uri(), paths, keys, nans


def files_opened(sql: str, tmp_path) -> int:
    con = duckdb.connect()
    con.execute("LOAD iceberg")
    profile = tmp_path / "profile.json"
    con.execute("PRAGMA enable_profiling = 'json'")
    con.execute(f"PRAGMA profiling_output = '{profile}'")
    con.execute("SET profiling_mode = 'detailed'")
    con.sql(sql).fetchall()
    con.execute("PRAGMA disable_profiling")

    def scans(node):
        if node.get("type") == "TABLE_SCAN":
            yield node
        for child in node.get("children", []):
            yield from scans(child)

    p = json.loads(profile.read_text())
    return sum(int(s["extra_info"].get("Total Files Read", 0))
               for op in p["operator"] for s in scans(op))


def test_register_table_lists_every_file(table_files):
    root_uri, paths, keys, nans = table_files
    metadata = register_table(root_uri, paths, keys, nans)
    assert metadata.startswith(f"{root_uri}/run/iceberg/metadata/")
    listed = duckdb.execute("LOAD iceberg").execute(
        f"SELECT file_path FROM iceberg_metadata('{metadata}') ORDER BY 1").fetchall()
    assert [r[0] for r in listed] == [f"{root_uri}/{k}" for k in keys]


def test_iceberg_scan_prunes_files(table_files, tmp_path):
    root_uri, paths, keys, nans = table_files
    metadata = register_table(root_uri, paths, keys, nans)
    source = f"iceberg_scan('{metadata}')"
    one_day = "time >= TIMESTAMP '2023-01-31' AND time < TIMESTAMP '2023-02-01'"  # file 3
    band = "longitude BETWEEN -135 AND -132 AND latitude BETWEEN -10 AND 10"  # file 4
    lower_only = "longitude >= -90"  # file 9; doubles prune on >= only with NaN counts
    for where in (one_day, band, lower_only):
        assert files_opened(f"SELECT count(*) FROM {source} WHERE {where}", tmp_path) == 1
    count = duckdb.execute("LOAD iceberg").execute(
        f"SELECT count(*) FROM {source} WHERE {one_day}").fetchall()[0][0]
    assert count == 100


def test_iceberg_schema_matches_output(table_files):
    root_uri, paths, keys, nans = table_files
    metadata = register_table(root_uri, paths, keys, nans)
    cols = duckdb.execute("LOAD iceberg").execute(
        f"DESCRIBE SELECT * FROM iceberg_scan('{metadata}')").fetchall()
    assert [c[0] for c in cols] == OUTPUT_SCHEMA.names


def test_finalize_report_has_iceberg_metadata(pipeline_run):  # noqa: F811
    store, root, paths, report = pipeline_run
    metadata = report["iceberg_metadata"]
    n = duckdb.execute("LOAD iceberg").execute(
        f"SELECT count(*) FROM iceberg_scan('{metadata}')").fetchall()[0][0]
    assert n == report["rows_out_compact"]


def test_nan_counts_recorded_in_manifest(table_files):
    root_uri, paths, keys, nans = table_files
    nans[keys[0]] = {**nans[keys[0]], "h_li": 4}
    table = StaticTable.from_metadata(register_table(root_uri, paths, keys, nans))
    schema = table.schema()
    files = table.inspect.files().to_pylist()
    by_path = {f["file_path"]: dict(f["nan_value_counts"]) for f in files}
    first = by_path[f"{root_uri}/{keys[0]}"]
    assert first[schema.find_field("h_li").field_id] == 4
    assert first[schema.find_field("longitude").field_id] == 0
    # Columns that were not counted make no claim.
    assert schema.find_field("time_ns").field_id not in first


def test_nan_counts_counts_float_columns_only():
    t = output_file(0, n=5)
    h = t["h_li"].to_pylist()
    t = t.set_column(t.schema.get_field_index("h_li"), "h_li",
                     pa.array([float("nan"), None, *h[2:]], pa.float32()))
    counts = nan_counts(t)
    assert counts["h_li"] == 1 and counts["longitude"] == 0
    assert set(counts) == {f.name for f in OUTPUT_SCHEMA if pa.types.is_floating(f.type)}
