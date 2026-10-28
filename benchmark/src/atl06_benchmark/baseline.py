"""Naive comparison layout: granule/time order, partitioned by month."""

import duckdb


def build_baseline(
    con: duckdb.DuckDBPyConnection, src_uri: str, dst_uri: str, row_group_size: int
) -> int:
    """Returns rows written."""
    source = f"read_parquet('{src_uri}')"
    con.execute(
        f"""
        COPY (SELECT *, month(time_ns) AS month FROM {source} ORDER BY srcid, time_ns)
        TO '{dst_uri}'
        (FORMAT parquet, PARTITION_BY (month), ROW_GROUP_SIZE {row_group_size},
         COMPRESSION zstd, OVERWRITE_OR_IGNORE)
        """
    )
    return con.execute(f"SELECT count(*) FROM {source}").fetchall()[0][0]
