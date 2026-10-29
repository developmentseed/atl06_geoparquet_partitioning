# ATL06 → GeoParquet with stac-hash partitioning — Design

Date: 2026-10-28
Status: Draft for review

## 1. Intent

We want to understand how well [stac-hash](https://www.gadom.ski/stac-hash/latest/usage/)
works as a partitioning/sort strategy for point-cloud style altimetry data, measured by
**predicate pushdown performance** on common queries. The test bed is **one year of
ICESat-2 ATL06** written as GeoParquet.

A separate, immature S2 DGGS project exists with a similar schema; we keep its column set
but do **not** need to match its layout, data release, or benchmarks. Field mappings should
emulate **SlideRule's ATL06 conventions as closely as possible**.

Two query shapes matter equally, and stac-hash was chosen because it may balance both:

1. **Small region, long time** — e.g. all points in a glacier basin over the whole year.
2. **Short time, large region** — e.g. a few days over an entire ice sheet.

## 2. Goals

- A Python library that streams ATL06 HDF5 granules, converts them to Arrow, computes a
  stac-hash per point, and writes sorted, range-partitioned GeoParquet.
- Balanced output files despite strongly non-uniform point density (track convergence near
  ±88°, surface-mask gaps over ocean, cloud gaps, pointing strategy).
- AWS infrastructure (Lambda + Step Functions, CDK) to run the full year at scale.
- A benchmark query harness that runs the two query shapes (and others) against the dataset
  and reports pushdown effectiveness.

## 2a. Definition of done

- Deployable code: the `processor` library, the CDK app (Lambda + Step Functions), and the
  benchmark harness, with tests that run locally without AWS.
- A **small spike run** (a few dozen granules) through the deployed pipeline, plus the
  benchmark harness run against its output, with results recorded — including the §9
  risk measurements (fragment fan-out, file-size spread, Lambda memory/runtime).
- The full-year run is **not** part of done. The user will supply Earthdata credentials
  when runs are kicked off; until then, nothing that needs EDL access is executed.

## 3. Non-goals

- Iceberg tables / catalogs (despite the repo name) — plain GeoParquet files plus a manifest.
- Forward processing of newly arriving granules; this is a one-off backfill of one year.
- Science-level filtering or QA of ATL06 (no dropping points by quality flag).
- Building or running the S2 comparison dataset.
- Region-specific (polar) hashers — noted as a possible later experiment only.

## 4. Anti-goals (would be a failure even if it "works")

- Badly skewed outputs: a few huge files and many near-empty ones.
- Field values that diverge from SlideRule's ATL06 conventions.
- Benchmark results that depend on custom query logic rather than engine pushdown.
- Millions of tiny intermediate S3 objects that make the shuffle step cost-dominated.
- Benchmarks that measure network noise rather than pushdown (e.g. no record of
  files/row groups/bytes touched, only wall-clock).

## 5. Data source

- **Product:** ATL06 (Land Ice Height), latest available release: **v007** (v006 has been retired from CMR; checked 2026-10-28).
- **Time span:** one full calendar year (default 2023; configurable).
- **Access:** NSIDC Earthdata cloud S3 bucket in **us-west-2**. All compute runs in
  us-west-2. Granule discovery via CMR.
- **Credentials:** Earthdata Login (EDL) username/password in AWS Secrets Manager. Lambdas
  exchange them for NSIDC temporary S3 credentials (~1 hour lifetime), cached per container
  and refreshed before expiry. Concurrency against EDL is capped to avoid throttling.

## 6. Output schema

Matches the comparison project, with `cell_id` (S2-specific) removed and `stac_hash` plus a
bbox covering column added:

| column | type | source (per beam group `gtXX/`) |
|---|---|---|
| time_ns | timestamp[ns] | `land_ice_segments/delta_time` + `ancillary_data/atlas_sdp_gps_epoch`, GPS → UTC |
| tide_ocean | float32 | `land_ice_segments/geophysical/tide_ocean` |
| region | uint8 | granule region number (from granule name / metadata) |
| gt | uint8 | beam id, SlideRule encoding (gt1l=10, gt1r=20, gt2l=30, gt2r=40, gt3l=50, gt3r=60) |
| n_fit_photons | int32 | `land_ice_segments/fit_statistics/n_fit_photons` |
| w_surface_window_final | float32 | `land_ice_segments/fit_statistics/w_surface_window_final` |
| atl06_quality_summary | int8 | `land_ice_segments/atl06_quality_summary` |
| rgt | uint16 | `orbit_info/rgt` |
| r_eff | float32 | `land_ice_segments/geophysical/r_eff` |
| seg_azimuth | float32 | `land_ice_segments/ground_track/seg_azimuth` |
| bsnow_h | float32 | `land_ice_segments/geophysical/bsnow_h` |
| h_robust_sprd | float32 | `land_ice_segments/fit_statistics/h_robust_sprd` |
| sigma_geo_h | float32 | `land_ice_segments/sigma_geo_h` |
| h_li_sigma | float32 | `land_ice_segments/h_li_sigma` |
| dh_fit_dx | float32 | `land_ice_segments/fit_statistics/dh_fit_dx` |
| x_atc | float64 | `land_ice_segments/ground_track/x_atc` |
| spot | uint8 | `gtXX` attribute `atlas_spot_number` |
| y_atc | float32 | `land_ice_segments/ground_track/y_atc` |
| srcid | int32 | index into the dataset's granule table (see §7.5) |
| bsnow_conf | int8 | `land_ice_segments/geophysical/bsnow_conf` |
| cycle | uint8 | `orbit_info/cycle_number` |
| h_li | float32 | `land_ice_segments/h_li` |
| segment_id | uint32 | `land_ice_segments/segment_id` |
| geometry | binary, `geoarrow.wkb` (EPSG:4326 / OGC:CRS84 PROJJSON) | point from `latitude`, `longitude` |
| **stac_hash** | uint64 | stac-hash of (time, lon, lat) — sort key |
| **bbox** | struct<xmin,ymin,xmax,ymax: float64> | GeoParquet 1.1 covering column |

Notes:
- Field mappings follow **SlideRule's ATL06 conventions** because the comparison schema
  appears to be SlideRule-derived. The comparison project's code is not available, so the
  mappings are built from the ATL06 data dictionary. Any mapping the builder can't confirm
  should be listed in the README as an assumption.
- Fill values (e.g. 3.4028235e38) are **mapped to null** — they would otherwise wreck
  min/max statistics on float columns. Rows are never dropped for quality reasons.
  *(Claude's call.)*
- GeoParquet metadata (`geo` key) declares `geometry` as primary column, point geometry type,
  and the `bbox` covering.

## 7. Library design (`processor` package)

Everything below runs locally on a single granule with no AWS dependency, so it is testable
in isolation. AWS-specific code (credentials, S3 paths, Lambda handlers) is a thin layer on top.

### 7.1 Granule reader (streaming)
- Opens an HDF5 granule via a file-like object (local path or S3 via fsspec/s3fs/obstore —
  builder's choice), reading only the datasets needed for the schema.
- Iterates **beam by beam** (six ground tracks), yielding Arrow `RecordBatch`es so memory is
  bounded by one beam's columns rather than the full granule.
- Beams missing from a granule (or empty) are skipped without error.

### 7.2 Arrow conversion
- Builds batches with the exact schema in §6, including WKB point geometry and bbox struct.

### 7.3 Hashing
- `stac_hash` computed with the stac-hash Python `Hasher`, **global bbox**, temporal extent
  = the configured year (start inclusive, end exclusive).
- Year membership is by each **point's own timestamp**. Points outside the year (granules
  straddling a boundary) are filtered out *before* hashing and counted in run stats; the
  batch API (`hash_all`) is then called on in-extent points only. Note `hash_all_clamped`
  pins rather than drops, so it must not be used for this. *(Claude's call.)*
- **What the hash implies for file shape:** bits interleave time, lat, lon (time most
  significant), 21 bits each. With N ≈ 2^k quantile files, each file covers roughly
  k/3 bits of each dimension — e.g. ~1000 files ≈ 16 time slices × 8 lat bands × 8 lon
  bands (≈3 weeks × 22.5° × 45° before density skew). Files are Z-order boxes, not pure
  time slices: a small-region/full-year query should touch ~one file per time slice plus
  boundary effects, then rely on row-group pruning within each. Measuring exactly this is
  the point of the benchmark.

### 7.4 Splitting (balanced hash ranges)
- A **split table** defines N contiguous `stac_hash` ranges, one per output file, chosen at
  row-count quantiles so every file has roughly the same row count.
- Produced from a sample of granules (§8 step 2). Target rows per file derived from target
  file size (default ~512 MB–1 GB compressed; configurable).
- The split table is persisted alongside the dataset and doubles as a file-level index.

### 7.5 Writing
- **Fragment writer** (convert phase): routes each row to its split range and writes one small
  Parquet fragment per (granule, range) touched, under `staging/range=NNNNN/`.
- **Compactor** (compact phase): reads all fragments for one range, sorts by `stac_hash`,
  writes one final GeoParquet file with:
  - row groups of a configurable size (default somewhere in 100k–1M rows; tuning this is part
    of the benchmark),
  - column statistics enabled (at minimum on `stac_hash`, `time_ns`, `bbox.*`),
  - compression: zstd (builder may tune).
- **Granule table**: `srcid → granule id/name` written as a small side table.

### 7.6 Config
Single typed settings object (year, product version, target file size, row group size,
sample fraction, concurrency limits, bucket/prefixes).

## 8. Pipeline (AWS)

Modeled on [developmentseed/virtualizarr-data-pipelines](https://github.com/developmentseed/virtualizarr-data-pipelines):
Python CDK app in `cdk/`, uv-managed, processor package built into a Docker-image Lambda.
Step Functions orchestrates:

1. **Inventory** (Lambda): query CMR for the year's ATL06 granules; write manifest + granule
   table (assigns `srcid`) to S3.
2. **Sample** (Lambda, possibly a small Distributed Map): hash a random sample of granules
   (default 1–2%), compute quantile split points, write the split table.
3. **Convert** (Distributed Map over the manifest; one or a few granules per Lambda):
   stream → Arrow → hash → route to fragments. Concurrency capped for EDL.
4. **Compact** (Distributed Map over split ranges; one range per Lambda): sort + write the
   final GeoParquet file. Target file size is bounded so a range fits in Lambda memory
   (≤10 GB).
5. **Finalize** (Lambda): write dataset manifest (files, hash ranges, row counts, bboxes,
   time ranges), validate total rows in == rows out, report per-file size distribution.

Failure handling: Distributed Map retries per item; converts and compactions are idempotent
(deterministic output keys, overwrite on retry).

S3 layout (illustrative):
```
s3://<bucket>/<run>/
  manifest/granules.parquet      # srcid → granule
  manifest/splits.parquet        # range id → [hash_lo, hash_hi)
  staging/range=NNNNN/<granule>.parquet
  data/part-NNNNN.parquet        # final GeoParquet
  manifest/dataset.parquet       # per-file stats
```

## 9. Risks to validate early (spike)

Before scaling, run the library locally/on a handful of Lambdas against a few dozen granules:

- **Fragment fan-out:** how many split ranges a single granule touches. Time is the most
  significant interleaved dimension, so a few-minute granule should touch a narrow set, but
  bit interleaving makes it hard to predict. If fan-out is large (hundreds per granule),
  batch multiple granules per convert Lambda and/or buffer fragments before writing.
- **Sample fidelity:** whether a 1–2% sample yields file sizes within an acceptable spread
  (e.g. max/median < 2×). Fallback: a full hash-histogram pass instead of sampling.
- **Lambda limits:** per-granule memory and runtime for convert; per-range memory for compact.

## 10. Benchmark harness

Part of this project. Runs a defined query suite against the final dataset (and, where
possible, a naive baseline such as the same data partitioned by granule or by month).

- **Queries** (parameterized, defined in a config file):
  - small region (e.g. a named glacier basin bbox) over the full year,
  - short time window (e.g. 1 day, 1 week) over a whole ice sheet (Antarctica, Greenland),
  - small region + short time,
  - point/segment lookup by rgt+cycle (optional).
- **Engines:** at least one Parquet reader that does row-group pruning against S3
  (e.g. DuckDB and/or PyArrow dataset). Builder chooses; report which.
- **No specialized query logic.** Queries are plain engine predicates on `time_ns` and the
  `bbox` covering struct fields (and/or the engine's native spatial filter), and pruning is
  entirely the engine's own predicate pushdown over file and row-group statistics. The
  `stac_hash` column is the *write-time* sort/partition key only; queries never reference it
  and the harness never translates queries into hash ranges. stac-hash's benefit shows up
  as tight per-file/per-row-group min/max on `time_ns` and `bbox`. *(User's decision.)*
- **Metrics recorded per query:** files touched, row groups read vs total, bytes fetched,
  rows returned, wall-clock time; results written as a table (CSV/Parquet) for comparison
  with the S2 project.
- Runs from an EC2 instance or Lambda in us-west-2 to keep network effects consistent.

## 11. Decided by the user

- Done = deployable code + small spike run; full-year run out of scope; EDL creds supplied
  by the user at run time.
- Hash-range files with hash-sorted rows inside; row groups provide finer pruning.
- Lambda (+ Step Functions where warranted), similar to virtualizarr-data-pipelines.
- Schema as listed, minus `cell_id`, plus `stac_hash` and a bbox covering column.
- ATL06 latest release, one calendar year (default 2023).
- Benchmark harness is in scope; it relies only on the engine's (DuckDB or other) predicate
  pushdown — no custom hash-range query logic.
- No need to match the S2 project beyond its column set; emulate SlideRule mappings closely.

## 12. Calls made by Claude (please check)

- SlideRule conventions for `gt`, `spot`, `srcid`, `region`; GPS → UTC for `time_ns`.
- Global hasher extent rather than polar-specific.
- Quantile split table from a sample, with full-histogram fallback.
- Default target file size ~512 MB–1 GB; row group size configurable.
- No quality filtering at write time.
- Naive baseline dataset in the benchmark (by granule or month) — optional but recommended.

## 13. Left to the builder

- HDF5 I/O library (h5py + fsspec, h5coro, etc.) and S3 client choice.
- Exact row-group size defaults and compression settings.
- Query engine(s) for the benchmark.
- Lambda memory sizing and batch sizes per Distributed Map item.
- Repo layout details beyond `processor/`, `cdk/`, `benchmark/`, `tests/`.

## 14. Amendment (2026-10-29): Iceberg table, bounded Compact, restart from Compact

Motivation: plain `read_parquet` over a glob opens every file footer on every query, and a
40,000-granule run failed in Compact with `Runtime.OutOfMemory` on one oversized range
(granule-clustered sampling leaves some ranges well above target).

**Decided by the user**
- Register the output files in an **Iceberg v2 table** (no data rewrite at registration) so
  DuckDB `iceberg_scan` prunes whole files from manifest bounds before opening footers.
- Output schema changes, applied in Compact (staging fragments keep the old schema, so an
  existing run's `staging/` can be reused):
  - `time_ns` becomes `int64` nanoseconds since the Unix epoch (SlideRule name and values).
    Reason: Iceberg v2 has no nanosecond timestamp, and pyiceberg rejects any
    `timestamp[ns]` column in a file even if the table schema omits it.
  - New `time: timestamp[us]` (naive UTC) immediately after `time_ns`; time filters use it.
  - New top-level `longitude`, `latitude: float64` immediately before `geometry`; spatial
    filters use them. Reason: pyiceberg writes no manifest bounds for struct leaves, so
    `bbox.*` cannot prune files. The `bbox` covering struct and GeoParquet metadata stay.
- A run can restart at Compact for an existing prefix (execution input
  `"from_step": "compact"`), reusing `staging/` and the split table.

**Calls made by Claude (please check)**
- Compact keeps memory bounded: fragments are downloaded to the Lambda's ephemeral disk,
  `stac_hash` alone is read to count rows, and a range above `max_rows_per_file`
  (default 30,000,000) is split at hash quantiles into contiguous sub-ranges, each written
  as its own file. Sorting uses `sort_indices` and writes one row group at a time, so peak
  memory is about one copy of a part plus one row group. Every file is now named
  `data/part-NNNNN-KK.parquet` (range, part); ranges that fit have one part, `KK = 00`.
- Iceberg metadata lives under `runs/<prefix>/iceberg/`, created by Finalize with pyiceberg's
  `SqlCatalog` (SQLite in `/tmp`, not persisted) and `add_files`. `report.json` records
  `iceberg_metadata` (the `metadata.json` URI); no Glue or S3 Tables catalog.
- Benchmark datasets gain `format = "parquet" | "iceberg"`; for Iceberg, `uri` is the
  `metadata.json`. Queries filter on `time`, `longitude`, `latitude` (still plain predicates,
  never `stac_hash`). The month baseline partitions by `month(time)`.

**Verified locally before planning** (pyiceberg 0.12.0, DuckDB 2.0.0, 20 synthetic files):
`iceberg_scan` opened 1/20 files for a 1-day `time` window and 0/20 for a `stac_hash` range;
it opened 20/20 for `bbox.*` predicates (no struct-leaf bounds). Not yet verified on S3.
