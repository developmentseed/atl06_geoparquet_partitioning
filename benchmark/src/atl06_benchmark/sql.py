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
