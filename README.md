# atl06_iceberg

Converts one year of ICESat-2 ATL06 into balanced, [stac-hash](https://www.gadom.ski/stac-hash/latest/usage/)
sorted GeoParquet, and benchmarks how well plain predicate pushdown prunes it.

Design: `docs/superpowers/specs/2026-10-28-atl06-stac-hash-geoparquet-design.md`

| Package | What it does |
|---|---|
| `processor/` (`atl06_processor`) | HDF5 → Arrow (beam by beam) → year filter + stac-hash → balanced hash-range files |
| `cdk/` (`atl06_cdk`) | Lambda + Step Functions pipeline (us-west-2) |
| `benchmark/` (`atl06_benchmark`) | DuckDB query suite, pushdown metrics, month-partitioned baseline |

## Pipeline

```
Inventory ─▶ Sample (Distributed Map) ─▶ PlanSplits ─▶ Convert (Distributed Map) ─▶ Compact (Distributed Map) ─▶ Finalize
 CMR search   hash a granule sample      quantile cuts   route rows to             sort one range,             report.json,
 srcid table                             → split table   staging/range=N/          write data/part-N-K.parquet Iceberg table
```

Each output file covers one contiguous `stac_hash` range, chosen at row-count quantiles so files hold
roughly equal row counts; rows inside are sorted by `stac_hash`. A range above
`max_rows_per_file` (default 25M, the 10 GB Compact Lambda's limit at ~330 B/row peak) is split at
hash quantiles into several contiguous files, `part-NNNNN-00`, `-01`, … `manifest/report.json`
records row balance, file-size spread, per-granule fragment fan-out, and `iceberg_metadata`.

Finalize registers every data file in an **Iceberg v2 table** under `runs/<prefix>/iceberg/`
(pyiceberg `add_files`; files are not rewritten; no catalog service). Readers open the
`metadata.json` named in `report.json`, e.g. DuckDB `iceberg_scan('<uri>')`, and skip whole files
from the manifest bounds before opening any footer.

Output columns differ from the staging fragments in three ways, all for Iceberg:
`time_ns` is `int64` nanoseconds (SlideRule's name and values; Iceberg v2 has no nanosecond
timestamp), `time` is a `timestamp[us]` for filtering, and top-level `longitude`/`latitude` duplicate
the point's bbox (Iceberg manifests carry no bounds for `bbox.*` struct fields). Compact also records
per-file NaN counts of float columns into the manifest; without them DuckDB cannot prune doubles on
`>=`.

## Setup and tests

```bash
./scripts/setup.sh   # uv sync, Node (nodeenv) + pinned AWS CDK CLI inside .venv
uv run pytest
uv run ruff check .
```

All tests run locally against synthetic ATL06-shaped HDF5 files; no AWS or Earthdata access needed.

## Deploy

1. Store Earthdata Login credentials:
   ```bash
   aws secretsmanager create-secret --region us-west-2 --name atl06/earthdata \
     --secret-string '{"username":"…","password":"…"}'
   ```
2. Configure and deploy (Docker must be running to build the Lambda image). Always use the CDK
   CLI from the venv via `uv run`, not a globally installed one:
   ```bash
   cp .env.sample .env        # set CDK_ACCOUNT etc.
   uv run --env-file .env cdk synth
   uv run --env-file .env cdk deploy
   ```
   Settings are `CDK_*` env vars (see `cdk/atl06_cdk/settings.py` and `.env.sample`), e.g.
   `CDK_TARGET_ROWS_PER_FILE`, `CDK_MAX_ROWS_PER_FILE`, `CDK_CONVERT_CONCURRENCY`, `CDK_BUCKET_NAME` (otherwise the stack
   creates a retained bucket).

## Spike run

```bash
aws stepfunctions start-execution --region us-west-2 \
  --state-machine-arn <PipelineArn> \
  --input '{"run_prefix": "runs/spike-1", "limit": 40}'
```

Execution input: `run_prefix` (required; single-use, so pass `"overwrite": true` to delete and
redo a previous run), optional `start`/`end` (ISO datetimes within `ATL06_YEAR`, default the whole
year) and `limit` (N granules evenly spaced through the year, so a spike spans all of it).

To redo only Compact and Finalize for a run that finished Convert (e.g. after a Compact failure or a
code change), start a new execution with `"from_step": "compact"`. It keeps `staging/` and the
split table, deletes `data/`, `stats/compact/` and `iceberg/`, and downloads nothing from NSIDC:

```bash
aws stepfunctions start-execution --region us-west-2 --state-machine-arn <PipelineArn> \
  --input '{"run_prefix": "runs/<prefix>", "from_step": "compact"}'
``` `.env.sample` already holds spike settings
(`CDK_SAMPLE_GRANULE_FRACTION=1.0`, a small `CDK_TARGET_ROWS_PER_FILE`) so a spike yields ≥ 10 files.

## Benchmark

Run from us-west-2 (EC2 or CloudShell) so network effects are consistent.

```bash
cp benchmark/queries.default.toml benchmark/queries.toml   # local, git-ignored; set the URIs
uv run atl06-bench baseline --src 's3://<bucket>/runs/<prefix>/data/*.parquet' \
  --dst 's3://<bucket>/runs/<prefix>/baseline' --row-group-size 500000   # optional
uv run atl06-bench run --suite benchmark/queries.toml --out results.parquet
```

Each dataset is `format = "parquet"` (a glob; DuckDB opens every footer) or `format = "iceberg"`
(`uri` = `report.json` `iceberg_metadata`; files are pruned from manifests first). Queries use only
plain predicates on `time`, `longitude` and `latitude`; pruning is DuckDB's own pushdown. Runs
written before these columns existed (`spike-1`, `spike-2`) cannot be queried with them; restart
them at Compact first.
Metrics per query and repeat: `files_total`, `footers_read` (files DuckDB actually opened: all of
them for a glob, only manifest survivors for Iceberg), `files_scanned` (files with at least one row
group whose statistics can match, derived from `parquet_metadata()`), row groups scanned/total,
bytes read, rows, wall time.

## Assumptions

- **SlideRule conventions**: `gt` gt1l=10 … gt3r=60; `spot` from the beam's `atlas_spot_number`
  attribute; `srcid` indexes `manifest/granules.parquet`; `rgt`, `cycle`, `region` from the granule name.
- **Time**: UTC, from `atlas_sdp_gps_epoch + delta_time` with a constant 18 s GPS−UTC offset
  (valid for all dates since 2017-01-01). `time_ns` is int64 nanoseconds since the Unix epoch;
  `time` is the same instant truncated to microseconds.
- **Fill values** (`_FillValue` attributes) become null; no rows are dropped for quality.
- **Year membership** is by each point's own timestamp, half-open `[Jan 1, Jan 1 next year)`.
- **Unhashable rows** (fill `latitude`, `longitude` or `delta_time`) are dropped and counted as
  `dropped_invalid` in `report.json`; a point with no time or position has no stac-hash.
- **Hasher**: global extent over the configured year.
- **Sampling**: 2% of granules, 1% of their rows, by default. The row estimate uses the realized
  granule fraction.
- **File size**: `target_rows_per_file` defaults to 10 M rows, pending measured bytes/row from the spike.
- **S3 access** is via [obstore](https://developmentseed.org/obstore/). Granules are read whole
  (one GET → `io.BytesIO` → h5py), so Lambda memory must exceed the largest granule. NSIDC
  credentials come from obstore's `NasaEarthdataCredentialProvider`, which refreshes them itself.
