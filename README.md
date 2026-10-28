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
 CMR search   hash a granule sample      quantile cuts   route rows to             sort one range,
 srcid table                             → split table   staging/range=N/          write data/part-N.parquet
```

Each output file covers one contiguous `stac_hash` range, chosen at row-count quantiles so files hold
roughly equal row counts; rows inside are sorted by `stac_hash`. `manifest/report.json` records
row balance, file-size spread, and per-granule fragment fan-out.

## Setup and tests

```bash
uv sync
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
2. Deploy (Docker required to build the Lambda image; the global `cdk` CLI must be ≥ 2.1146, so use `npx`):
   ```bash
   cd cdk
   CDK_ACCOUNT=<account-id> npx aws-cdk@latest deploy
   ```
   Settings are `CDK_*` env vars (see `cdk/atl06_cdk/settings.py`), e.g. `CDK_TARGET_ROWS_PER_FILE`,
   `CDK_CONVERT_CONCURRENCY`, `CDK_BUCKET_NAME` (otherwise the stack creates a retained bucket).

## Spike run

```bash
aws stepfunctions start-execution --region us-west-2 \
  --state-machine-arn <PipelineArn> \
  --input '{"run_prefix": "runs/spike-1", "limit": 40}'
```

Execution input: `run_prefix` (required), optional `start`/`end` (ISO datetimes, default the
configured year) and `limit` (first N granules by name). For a spike, deploy with
`CDK_SAMPLE_GRANULE_FRACTION=1.0` and a small `CDK_TARGET_ROWS_PER_FILE` so you get ≥ 10 files.

## Benchmark

Run from us-west-2 (EC2 or CloudShell) so network effects are consistent.

```bash
uv run atl06-bench baseline --src 's3://<bucket>/runs/spike-1/data/*.parquet' \
  --dst 's3://<bucket>/runs/spike-1/baseline' --row-group-size 500000
# edit benchmark/queries.toml dataset URIs, then:
uv run atl06-bench run --suite benchmark/queries.toml --out results.parquet
```

Queries use only plain predicates on `time_ns` and `bbox.*`; pruning is DuckDB's own pushdown.
Metrics per query and repeat: files read/total, row groups scanned/total, bytes read, rows, wall time.

## Assumptions

- **SlideRule conventions**: `gt` gt1l=10 … gt3r=60; `spot` from the beam's `atlas_spot_number`
  attribute; `srcid` indexes `manifest/granules.parquet`; `rgt`, `cycle`, `region` from the granule name.
- **Time**: `time_ns` = UTC, from `atlas_sdp_gps_epoch + delta_time` with a constant 18 s GPS−UTC
  offset (valid for all dates since 2017-01-01).
- **Fill values** (`_FillValue` attributes) become null; no rows are dropped for quality.
- **Year membership** is by each point's own timestamp, half-open `[Jan 1, Jan 1 next year)`.
- **Hasher**: global extent over the configured year.
- **Sampling**: 2% of granules, 1% of their rows, by default. The row estimate uses the realized
  granule fraction.
- **File size**: `target_rows_per_file` defaults to 10 M rows, pending measured bytes/row from the spike.
- **Granule reads**: the NSIDC filesystem reads each granule whole (`default_cache_type="all"`), so
  Lambda memory must exceed the largest granule.
