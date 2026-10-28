"""Plain-SQL queries. Pruning is left entirely to the engine's predicate pushdown."""

from atl06_benchmark.config import Query


def build_sql(dataset_uri: str, q: Query) -> str:
    predicates = []
    if q.bbox is not None:
        minx, miny, maxx, maxy = q.bbox
        predicates += [
            f"bbox.xmin >= {minx}",
            f"bbox.xmax <= {maxx}",
            f"bbox.ymin >= {miny}",
            f"bbox.ymax <= {maxy}",
        ]
    if q.start is not None:
        predicates.append(f"time_ns >= TIMESTAMP '{q.start}'")
    if q.end is not None:
        predicates.append(f"time_ns < TIMESTAMP '{q.end}'")
    sql = f"SELECT {', '.join(q.columns)} FROM read_parquet('{dataset_uri}')"
    return f"{sql} WHERE {' AND '.join(predicates)}" if predicates else sql


def build_files_scanned_sql(dataset_uri: str, q: Query) -> str:
    """Count files with at least one row group whose min/max statistics could satisfy
    ``q``'s predicates. A metric only: derived from Parquet metadata, never used to
    rewrite or route the benchmark query itself."""
    conditions = []
    if q.bbox is not None:
        minx, miny, maxx, maxy = q.bbox
        conditions += [
            f"max(CASE WHEN col = 'bbox, xmin' THEN hi END)::DOUBLE >= {minx}",
            f"min(CASE WHEN col = 'bbox, xmax' THEN lo END)::DOUBLE <= {maxx}",
            f"max(CASE WHEN col = 'bbox, ymin' THEN hi END)::DOUBLE >= {miny}",
            f"min(CASE WHEN col = 'bbox, ymax' THEN lo END)::DOUBLE <= {maxy}",
        ]
    if q.start is not None:
        conditions.append(
            f"max(CASE WHEN col = 'time_ns' THEN hi END)::TIMESTAMP_NS >= TIMESTAMP '{q.start}'"
        )
    if q.end is not None:
        conditions.append(
            f"min(CASE WHEN col = 'time_ns' THEN lo END)::TIMESTAMP_NS < TIMESTAMP '{q.end}'"
        )
    having = f"HAVING {' AND '.join(conditions)}" if conditions else ""
    return f"""
        SELECT count(DISTINCT file_name) FROM (
            SELECT file_name, row_group_id
            FROM (
                SELECT file_name, row_group_id, path_in_schema AS col,
                       stats_min_value AS lo, stats_max_value AS hi
                FROM parquet_metadata('{dataset_uri}')
            )
            GROUP BY file_name, row_group_id
            {having}
        )
    """
