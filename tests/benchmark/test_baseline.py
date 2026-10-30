import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from atl06_benchmark.baseline import build_baseline
from atl06_benchmark.cli import main


def write_source(path, n=5_000):
    t = pa.array(np.linspace(1_672_531_200, 1_704_067_199, n).astype("int64") * 10**6,
                 pa.timestamp("us"))
    pq.write_table(pa.table({"time": t, "srcid": np.arange(n, dtype=np.int32) % 7,
                             "h_li": np.ones(n, np.float32)}), path)


def test_baseline_roundtrip_rowcount(tmp_path):
    write_source(tmp_path / "src.parquet")
    dst = tmp_path / "baseline"
    n = build_baseline(duckdb.connect(), str(tmp_path / "src.parquet"), str(dst), 1000)
    assert n == 5_000
    assert len(list(dst.glob("month=*"))) == 12
    sql = f"SELECT count(*) FROM read_parquet('{dst}/**/*.parquet')"
    assert duckdb.sql(sql).fetchall()[0][0] == 5_000


def test_cli_baseline(tmp_path):
    write_source(tmp_path / "src.parquet")
    main(["baseline", "--src", str(tmp_path / "src.parquet"), "--dst", str(tmp_path / "b"),
          "--row-group-size", "1000"])
    assert any((tmp_path / "b").glob("month=*/*.parquet"))
