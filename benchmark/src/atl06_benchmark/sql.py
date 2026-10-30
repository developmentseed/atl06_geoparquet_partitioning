"""Plain-SQL queries. Pruning is left entirely to the engine's predicate pushdown."""

from atl06_benchmark.config import Dataset, Query


def source_sql(ds: Dataset) -> str:
    if ds.format == "iceberg":
        return f"iceberg_scan('{ds.uri}')"
    return f"read_parquet('{ds.uri}')"


def data_files_sql(ds: Dataset) -> str:
    """The dataset's data files, one per row."""
    if ds.format == "iceberg":
        return (
            f"SELECT file_path FROM iceberg_metadata('{ds.uri}') "
            "WHERE content = 'DATA' AND status != 'DELETED' ORDER BY 1"
        )
    return f"SELECT file FROM glob('{ds.uri}') ORDER BY 1"


def build_sql(ds: Dataset, q: Query) -> str:
    # Points: longitude/latitude equal the bbox, and unlike bbox.* they have Iceberg bounds.
    predicates = []
    if q.bbox is not None:
        minx, miny, maxx, maxy = q.bbox
        predicates += [
            f"longitude >= {minx}",
            f"longitude <= {maxx}",
            f"latitude >= {miny}",
            f"latitude <= {maxy}",
        ]
    if q.start is not None:
        predicates.append(f"time >= TIMESTAMP '{q.start}'")
    if q.end is not None:
        predicates.append(f"time < TIMESTAMP '{q.end}'")
    sql = f"SELECT {', '.join(q.columns)} FROM {source_sql(ds)}"
    return f"{sql} WHERE {' AND '.join(predicates)}" if predicates else sql


def build_files_scanned_sql(files: list[str], q: Query) -> str:
    """Count files with at least one row group whose min/max statistics could satisfy
    ``q``'s predicates. A metric only: derived from Parquet metadata, never used to
    rewrite or route the benchmark query itself."""
    conditions = []
    if q.bbox is not None:
        minx, miny, maxx, maxy = q.bbox
        conditions += [
            f"max(CASE WHEN col = 'longitude' THEN hi END)::DOUBLE >= {minx}",
            f"min(CASE WHEN col = 'longitude' THEN lo END)::DOUBLE <= {maxx}",
            f"max(CASE WHEN col = 'latitude' THEN hi END)::DOUBLE >= {miny}",
            f"min(CASE WHEN col = 'latitude' THEN lo END)::DOUBLE <= {maxy}",
        ]
    if q.start is not None:
        conditions.append(
            f"max(CASE WHEN col = 'time' THEN hi END)::TIMESTAMP >= TIMESTAMP '{q.start}'"
        )
    if q.end is not None:
        conditions.append(
            f"min(CASE WHEN col = 'time' THEN lo END)::TIMESTAMP < TIMESTAMP '{q.end}'"
        )
    having = f"HAVING {' AND '.join(conditions)}" if conditions else ""
    file_list = ", ".join(f"'{f}'" for f in files)
    return f"""
        SELECT count(DISTINCT file_name) FROM (
            SELECT file_name, row_group_id
            FROM (
                SELECT file_name, row_group_id, path_in_schema AS col,
                       stats_min_value AS lo, stats_max_value AS hi
                FROM parquet_metadata([{file_list}])
            )
            GROUP BY file_name, row_group_id
            {having}
        )
    """
