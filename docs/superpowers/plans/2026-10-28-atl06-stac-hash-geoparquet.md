# ATL06 stac-hash GeoParquet Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a library + AWS pipeline that converts one year of ICESat-2 ATL06 HDF5 into balanced, stac-hash–sorted GeoParquet files, plus a DuckDB benchmark harness that measures predicate-pushdown effectiveness; prove it with a small spike run.

**Architecture:** A pure-Python `atl06_processor` package does all data work against an injected fsspec filesystem (local in tests, S3 in Lambda). Five thin Lambda handlers wrap it and a Step Functions state machine (CDK, Python) chains Inventory → Sample (Distributed Map) → PlanSplits → Convert (Distributed Map) → Compact (Distributed Map) → Finalize. A separate `atl06_benchmark` package runs plain-SQL DuckDB queries and reads pruning metrics from DuckDB's JSON profiler.

**Tech Stack:** Python 3.12, uv workspace, pyarrow ≥21, numpy, h5py, s3fs, earthaccess, stac-hash 0.0.2, pyproj, pydantic-settings, DuckDB ≥2.0, AWS CDK v2 (Python), pytest, ruff.

**Spec:** `docs/superpowers/specs/2026-10-28-atl06-stac-hash-geoparquet-design.md`

## Global Constraints

- Output column set and order exactly as spec §6, minus `cell_id`, plus `stac_hash: uint64` and `bbox: struct<xmin,ymin,xmax,ymax: float64>` appended after `geometry`.
- SlideRule conventions: `gt` gt1l=10, gt1r=20, gt2l=30, gt2r=40, gt3l=50, gt3r=60; `spot` from the beam group's `atlas_spot_number` attribute; `time_ns` = UTC nanoseconds since Unix epoch; `srcid` = index into the run's granule table.
- Fill values (`_FillValue` attribute on a dataset) → null. Rows are never dropped for quality.
- Year membership by each point's own `time_ns` in `[Jan 1 year, Jan 1 year+1)` UTC; out-of-year points are filtered **before** hashing and counted. Never use `hash_all_clamped`/`hash_clamped`.
- Hasher: `stac_hash.Hasher(start, end)` with tz-aware UTC datetimes, global extent (no bbox). Verified: `end` is *inclusive* in the library and hashes are < 2**63.
- Default year 2023, ATL06 version `"006"` (configurable).
- Output files: one per contiguous hash range, rows sorted by `stac_hash`, zstd, statistics enabled, configurable row-group size; GeoParquet 1.1 `geo` metadata with `bbox` covering.
- Benchmark uses only plain predicates on `time_ns` and `bbox.*`; queries never reference `stac_hash`.
- Nothing that needs Earthdata access runs without the user supplying credentials.
- All AWS resources in `us-west-2`.

## Review Focus

1. A granule straddling the year boundary → in-year points kept, out-of-year points dropped and counted in stats (not clamped onto the boundary). Test: Task 3 `test_filter_and_hash_drops_out_of_year`.
2. A granule with a missing beam group, or a beam with zero segments → that beam is skipped, others still convert. Test: Task 2 `test_missing_and_empty_beams_skipped`.
3. Integer and float fill values → null, and column statistics on `h_li` don't contain 3.4028235e38. Test: Task 2 `test_fill_values_become_null` and Task 6 `test_stats_exclude_fill_values`.
4. A split range that receives no rows in the full run (sample said it would), and duplicate quantile cut points → no empty output files, no crash, finalize still balances row counts. Tests: Task 4 `test_duplicate_cuts_deduped`, Task 6 `test_compact_range_with_no_fragments_returns_none`.
5. A Lambda retry re-running convert for the same granule → fragment keys are deterministic and overwritten, so no duplicated rows. Test: Task 5 `test_convert_is_idempotent`.

---

## File Structure

```
pyproject.toml                        # uv workspace root (members: processor, benchmark, cdk); dev deps
README.md                             # setup, deploy, spike run, assumptions list
processor/
  pyproject.toml                      # package atl06_processor
  Dockerfile                          # Lambda image (public.ecr.aws/lambda/python:3.12)
  src/atl06_processor/
    __init__.py
    config.py        # Settings (pydantic-settings, env prefix ATL06_)
    schema.py        # SCHEMA, GT_CODES, CRS84_PROJJSON, geo_metadata()
    granule.py       # GranuleInfo + parse_granule_name()
    reader.py        # iter_beam_batches(): HDF5 → Arrow, beam by beam
    geometry.py      # points_to_wkb(), bbox_struct()
    hashing.py       # year_bounds(), make_hasher(), filter_and_hash()
    splits.py        # SplitTable: from_sample(), range_ids(), save()/load()
    sample.py        # sample_granule(): row count + subsampled hashes
    fragments.py     # convert_granule(): route rows to staging fragments
    compact.py       # compact_range(): sort + write final GeoParquet
    finalize.py      # finalize_run(): dataset manifest + validation report
    inventory.py     # build_inventory(): CMR search → granule table + sample list
    credentials.py   # NsidcS3: cached, auto-refreshing s3fs for NSIDC
    paths.py         # RunPaths: every S3 key layout in one place
    handlers.py      # Lambda handlers (thin)
benchmark/
  pyproject.toml                      # package atl06_benchmark
  queries.toml                        # default query suite
  src/atl06_benchmark/
    __init__.py
    config.py        # load_suite(): parse queries.toml
    sql.py           # build_sql()
    runner.py        # QueryMetrics, run_query(), run_suite()
    baseline.py      # build_baseline(): time-ordered, month-partitioned copy
    cli.py           # `atl06-bench run|baseline`
cdk/
  pyproject.toml
  cdk.json
  app.py
  atl06_cdk/
    __init__.py
    settings.py      # StackSettings (pydantic-settings, env prefix CDK_)
    stack.py         # Atl06StacHashStack
tests/
  conftest.py                         # synthetic ATL06 granule factory fixture
  processor/test_*.py
  benchmark/test_*.py
  cdk/test_stack.py
docs/spike-results.md                 # written in Task 12
```

---

### Task 1: Workspace scaffolding, config, schema

**Files:**
- Create: `pyproject.toml`, `processor/pyproject.toml`, `processor/src/atl06_processor/{__init__,config,schema}.py`, `tests/processor/test_schema.py`, `.gitignore`

**Interfaces:**
- Produces:
  - `Settings` (pydantic `BaseSettings`, env prefix `ATL06_`): `year: int = 2023`, `short_name: str = "ATL06"`, `version: str = "006"`, `target_rows_per_file: int = 10_000_000`, `row_group_size: int = 500_000`, `sample_granule_fraction: float = 0.02`, `sample_row_fraction: float = 0.01`, `bucket: str = ""`, `run_prefix: str = "runs/dev"`, `edl_secret_name: str = "atl06/earthdata"`.
  - `SCHEMA: pa.Schema` — columns/types/order per Global Constraints; `geometry` field metadata `ARROW:extension:name = "geoarrow.wkb"` and `ARROW:extension:metadata = json.dumps({"crs": CRS84_PROJJSON})`.
  - `GT_CODES: dict[str, int]` (`"gt1l": 10` … `"gt3r": 60`).
  - `CRS84_PROJJSON: dict` from `pyproj.CRS("OGC:CRS84").to_json_dict()`.
  - `geo_metadata(bbox: tuple[float,float,float,float]) -> dict` — GeoParquet 1.1: `version "1.1.0"`, `primary_column "geometry"`, column `geometry` with `encoding "WKB"`, `geometry_types ["Point"]`, `crs`, `bbox`, `covering {"bbox": {"xmin": ["bbox","xmin"], "ymin": ["bbox","ymin"], "xmax": ["bbox","xmax"], "ymax": ["bbox","ymax"]}}`.

- [ ] **Step 1:** Create uv workspace root (`[tool.uv.workspace] members = ["processor", "benchmark", "cdk"]`, dev group: pytest, ruff, duckdb) and `processor/pyproject.toml` with deps `pyarrow>=21`, `numpy`, `h5py`, `s3fs`, `earthaccess`, `stac-hash==0.0.2`, `pyproj`, `pydantic-settings`. Create empty-but-valid `benchmark/` and `cdk/` pyprojects so `uv sync` works (filled in later tasks). Python `>=3.12`.
- [ ] **Step 2: Write failing tests** in `tests/processor/test_schema.py`:

```python
def test_schema_column_order():
    assert SCHEMA.names == ["time_ns","tide_ocean","region","gt","n_fit_photons",
        "w_surface_window_final","atl06_quality_summary","rgt","r_eff","seg_azimuth",
        "bsnow_h","h_robust_sprd","sigma_geo_h","h_li_sigma","dh_fit_dx","x_atc","spot",
        "y_atc","srcid","bsnow_conf","cycle","h_li","segment_id","geometry","stac_hash","bbox"]

def test_schema_types():
    assert SCHEMA.field("time_ns").type == pa.timestamp("ns")
    assert SCHEMA.field("x_atc").type == pa.float64()
    assert SCHEMA.field("h_li").type == pa.float32()
    assert SCHEMA.field("rgt").type == pa.uint16()
    assert SCHEMA.field("segment_id").type == pa.uint32()
    assert SCHEMA.field("stac_hash").type == pa.uint64()
    assert SCHEMA.field("bbox").type == pa.struct([(k, pa.float64()) for k in ("xmin","ymin","xmax","ymax")])
    assert SCHEMA.field("geometry").metadata[b"ARROW:extension:name"] == b"geoarrow.wkb"

def test_gt_codes():
    assert GT_CODES == {"gt1l":10,"gt1r":20,"gt2l":30,"gt2r":40,"gt3l":50,"gt3r":60}

def test_geo_metadata_has_covering():
    md = geo_metadata((-1.0, -2.0, 3.0, 4.0))
    col = md["columns"]["geometry"]
    assert md["version"] == "1.1.0" and col["encoding"] == "WKB"
    assert col["covering"]["bbox"]["xmin"] == ["bbox", "xmin"]
    assert col["bbox"] == [-1.0, -2.0, 3.0, 4.0]
```

- [ ] **Step 3:** Run `uv run pytest tests/processor/test_schema.py -v` → FAIL (import error).
- [ ] **Step 4:** Implement `config.py` and `schema.py` per Interfaces.
- [ ] **Step 5:** Run tests → PASS. Run `uv run ruff check .` → clean.
- [ ] **Step 6:** Commit `feat: workspace scaffolding, settings, output schema`.

---

### Task 2: Synthetic granule fixture, granule names, HDF5 reader, geometry

**Files:**
- Create: `tests/conftest.py`, `processor/src/atl06_processor/{granule,reader,geometry}.py`, `tests/processor/test_reader.py`, `tests/processor/test_geometry.py`

**Interfaces:**
- Consumes: `SCHEMA`, `GT_CODES` (Task 1).
- Produces:
  - `GranuleInfo(name: str, rgt: int, cycle: int, region: int)` frozen dataclass; `parse_granule_name(name: str) -> GranuleInfo` for `ATL06_YYYYMMDDhhmmss_ttttccrr_vvv_ee.h5` (`tttt`=rgt, `cc`=cycle, `rr`=region).
  - `points_to_wkb(lon: np.ndarray, lat: np.ndarray) -> pa.BinaryArray` — little-endian WKB Point (21 bytes each), built with numpy (no shapely).
  - `bbox_struct(lon, lat) -> pa.StructArray` (xmin=xmax=lon, ymin=ymax=lat).
  - `iter_beam_batches(h5: h5py.File, info: GranuleInfo, srcid: int) -> Iterator[pa.RecordBatch]` — one batch per present, non-empty beam; schema = `PRE_HASH_SCHEMA` (add to `schema.py`: `SCHEMA` without `stac_hash`, which Task 3 adds).
  - Test fixture `make_granule(tmp_path, name, beams: dict[str, BeamSpec]) -> Path` in `tests/conftest.py` that writes an ATL06-shaped HDF5 file: `ancillary_data/atlas_sdp_gps_epoch` (= 1198800018.0), per beam `gtXX` group with attr `atlas_spot_number` (bytes, e.g. `b"3"`) and `land_ice_segments/{delta_time,latitude,longitude,h_li,h_li_sigma,atl06_quality_summary,segment_id,sigma_geo_h}`, `.../fit_statistics/{n_fit_photons,w_surface_window_final,h_robust_sprd,dh_fit_dx}`, `.../ground_track/{x_atc,y_atc,seg_azimuth}`, `.../geophysical/{tide_ocean,bsnow_h,bsnow_conf,r_eff}`; float datasets carry `_FillValue = np.float32(3.4028235e38)`, `bsnow_conf` `_FillValue = np.int8(127)`. `BeamSpec` lets tests set `delta_time`, `lat`, `lon`, and inject fill values.

Mapping (spec §6) — `delta_time` → `time_ns`:

```
gps_seconds = atlas_sdp_gps_epoch + delta_time      # seconds since GPS epoch 1980-01-06T00:00:00
utc_ns      = (GPS_EPOCH_UNIX_S + gps_seconds - GPS_UTC_LEAP_S) * 1e9   # GPS_EPOCH_UNIX_S = 315964800, GPS_UTC_LEAP_S = 18
```
Do the arithmetic in int64 nanoseconds after rounding `gps_seconds * 1e9` to avoid float precision loss (sub-µs error is acceptable). `GPS_UTC_LEAP_S = 18` is valid for all dates since 2017-01-01; add a comment saying so.

- [ ] **Step 1: Write failing tests:**

```python
def test_parse_granule_name():
    assert parse_granule_name("ATL06_20230105123456_01790201_006_01.h5") == GranuleInfo(
        "ATL06_20230105123456_01790201_006_01.h5", rgt=179, cycle=2, region=1)

def test_beam_batches_one_per_beam(make_granule, tmp_path): ...
    # two beams of 5 segments → 2 batches, gt values {10, 40}, spot parsed from attr, srcid/rgt/cycle/region constant

def test_time_ns_is_utc(make_granule, tmp_path):
    # delta_time chosen so gps time == 2023-03-01T00:00:18 GPS → time_ns == 2023-03-01T00:00:00 UTC

def test_fill_values_become_null(make_granule, tmp_path):
    # h_li[2] = 3.4028235e38, bsnow_conf[1] = 127 → those cells are null; others unchanged

def test_missing_and_empty_beams_skipped(make_granule, tmp_path):
    # gt1l absent, gt2l present with 0 segments, gt3r with 3 → exactly one batch, gt == 60

def test_points_to_wkb_roundtrip():
    wkb = points_to_wkb(np.array([10.5]), np.array([-70.25]))
    assert wkb[0].as_py() == struct.pack("<BIdd", 1, 1, 10.5, -70.25)
```

- [ ] **Step 2:** Run `uv run pytest tests/processor/test_reader.py tests/processor/test_geometry.py -v` → FAIL.
- [ ] **Step 3:** Implement `granule.py`, `geometry.py`, `reader.py`. Reader reads each needed dataset once per beam with `h5[path][()]`, masks `_FillValue` via `pa.array(values, mask=values == fill)`, casts to schema types, and builds `geometry`/`bbox` from `longitude`/`latitude`.
- [ ] **Step 4:** Run tests → PASS.
- [ ] **Step 5:** Commit `feat: ATL06 HDF5 beam reader with SlideRule mappings`.

---

### Task 3: Year filter + stac-hash

**Files:**
- Create: `processor/src/atl06_processor/hashing.py`, `tests/processor/test_hashing.py`

**Interfaces:**
- Consumes: `PRE_HASH_SCHEMA` batches (Task 2), `SCHEMA` (Task 1).
- Produces:
  - `year_bounds(year: int) -> tuple[datetime, datetime]` — tz-aware UTC `[Jan 1 year, Jan 1 year+1)`.
  - `make_hasher(year: int) -> stac_hash.Hasher` — `Hasher(start, end)`, global.
  - `filter_and_hash(batch: pa.RecordBatch, year: int, hasher: Hasher) -> tuple[pa.RecordBatch, int]` — returns a batch conforming to `SCHEMA` (with `stac_hash` inserted before `bbox`) and the count of rows dropped as out-of-year. Filter with `pc.greater_equal`/`pc.less` on `time_ns`; convert remaining `time_ns` to tz-aware datetimes via `.cast(pa.timestamp("us", tz="UTC")).to_pylist()`; call `hasher.hash_all(dts, lons, lats)` (default `skip_invalid=False`, so any surprise out-of-extent value raises loudly).

- [ ] **Step 1: Write failing tests:**

```python
def test_filter_and_hash_drops_out_of_year():
    # 3 rows: 2022-12-31T23:59:59, 2023-06-01, 2024-01-01T00:00:00 → 1 row kept, dropped == 2

def test_hash_matches_library():
    # kept row's stac_hash == make_hasher(2023).hash(dt, lon, lat)

def test_output_conforms_to_schema():
    assert out.schema.names == SCHEMA.names and out.schema.field("stac_hash").type == pa.uint64()

def test_all_rows_out_of_year_returns_empty_batch():
    # len(out) == 0, dropped == n, no exception
```

- [ ] **Step 2:** Run → FAIL. **Step 3:** Implement. **Step 4:** Run → PASS.
- [ ] **Step 5:** Commit `feat: year filter and stac-hash column`.

---

### Task 4: Split table and sampling

**Files:**
- Create: `processor/src/atl06_processor/{splits,sample,paths}.py`, `tests/processor/test_splits.py`, `tests/processor/test_sample.py`

**Interfaces:**
- Consumes: `iter_beam_batches`, `filter_and_hash`, `parse_granule_name`.
- Produces:
  - `RunPaths(root: str)` dataclass with methods returning keys: `granules_table()` → `{root}/manifest/granules.parquet`, `granules_json()` → `manifest/granules.json`, `sample_json()` → `manifest/sample.json`, `sample_part(granule)` → `sample/{stem}.parquet`, `splits()` → `manifest/splits.parquet`, `ranges_json()` → `manifest/ranges.json`, `fragment(range_id, granule)` → `staging/range={range_id:05d}/{stem}.parquet`, `fragment_dir(range_id)`, `convert_stats(granule)` → `stats/convert/{stem}.json`, `data_file(range_id)` → `data/part-{range_id:05d}.parquet`, `compact_stats(range_id)` → `stats/compact/{range_id:05d}.json`, `dataset_manifest()` → `manifest/dataset.parquet`, `report()` → `manifest/report.json`.
  - `sample_granule(fs, src_path: str, info: GranuleInfo, year: int, row_fraction: float, seed: int) -> tuple[int, np.ndarray]` — total in-year row count and a uniform random subsample (`np.random.default_rng(seed)`) of `stac_hash` values (uint64). Writes nothing; handler persists.
  - `SplitTable(cuts: np.ndarray[uint64])` — interior cut points, strictly increasing. `n_ranges = len(cuts)+1`. Range `i` = `[lo_i, hi_i)` with `lo_0 = 0`, `hi_last = 2**63`.
    - `SplitTable.from_sample(hashes: np.ndarray, est_total_rows: int, target_rows_per_file: int) -> SplitTable`: `n = max(1, ceil(est_total_rows / target_rows_per_file))`; cuts = `np.quantile(hashes, [k/n for k in 1..n-1], method="inverted_cdf")` as uint64, then `np.unique` (dedupes) and drop any `0`.
    - `range_ids(hashes: np.ndarray) -> np.ndarray[int32]` = `np.searchsorted(cuts, hashes, side="right")`.
    - `ranges() -> list[tuple[int, int, int]]` (range_id, lo, hi).
    - `save(fs, path)` / `load(fs, path)` via a Parquet table `(range_id int32, hash_lo uint64, hash_hi uint64)`.
  - Estimate used by PlanSplits: `est_total_rows = sum(granule_row_counts) / sample_granule_fraction`.

- [ ] **Step 1: Write failing tests:**

```python
def test_from_sample_balanced():
    h = np.random.default_rng(0).integers(0, 2**63, 100_000, dtype=np.uint64)
    st = SplitTable.from_sample(h, est_total_rows=1_000_000, target_rows_per_file=100_000)
    assert st.n_ranges == 10
    counts = np.bincount(st.range_ids(h)); assert counts.max() / counts.min() < 1.1

def test_skewed_sample_still_balanced():
    # 90% of hashes in a narrow band → bincount max/min < 1.2

def test_duplicate_cuts_deduped():
    st = SplitTable.from_sample(np.full(1000, 42, np.uint64), 1000, 100)
    assert np.all(np.diff(st.cuts.astype(np.int64)) > 0)

def test_value_on_cut_goes_right():
    st = SplitTable(np.array([100, 200], np.uint64))
    assert list(st.range_ids(np.array([99, 100, 199, 200], np.uint64))) == [0, 1, 1, 2]

def test_save_load_roundtrip(tmp_path): ...

def test_sample_granule_counts_and_subsamples(make_granule, tmp_path):
    # 1000 in-year rows, row_fraction 0.1 → count == 1000, 50 <= len(sample) <= 150
```

- [ ] **Step 2:** Run → FAIL. **Step 3:** Implement. **Step 4:** Run → PASS.
- [ ] **Step 5:** Commit `feat: quantile split table and granule sampling`.

---

### Task 5: Convert — route rows to staging fragments

**Files:**
- Create: `processor/src/atl06_processor/fragments.py`, `tests/processor/test_fragments.py`

**Interfaces:**
- Consumes: `iter_beam_batches`, `filter_and_hash`, `SplitTable`, `RunPaths`.
- Produces:
  - `ConvertStats` dataclass: `granule: str, rows_in: int, rows_out: int, dropped_out_of_year: int, ranges_touched: int, fragment_rows: dict[int, int]`; `.to_json()`/`from_json()`.
  - `convert_granule(src_fs, src_path: str, out_fs, paths: RunPaths, info: GranuleInfo, srcid: int, year: int, splits: SplitTable) -> ConvertStats` — streams beams, hashes, groups rows by `splits.range_ids`, accumulates per-range tables for this granule, writes one fragment per touched range at `paths.fragment(range_id, info.name)` (zstd, no geo metadata needed), writes stats JSON to `paths.convert_stats(info.name)`. Deterministic keys = idempotent retries.

- [ ] **Step 1: Write failing tests** (local `fsspec.filesystem("file")`, root under `tmp_path`):

```python
def test_convert_routes_rows_by_range(make_granule, tmp_path):
    # every row in fragment range=r has splits.range_ids(stac_hash) == r; sum of fragment rows == stats.rows_out

def test_convert_records_fanout(make_granule, tmp_path):
    assert stats.ranges_touched == len(stats.fragment_rows)

def test_convert_is_idempotent(make_granule, tmp_path):
    # run twice → same file list, total fragment rows unchanged
```

- [ ] **Step 2:** Run → FAIL. **Step 3:** Implement. **Step 4:** Run → PASS.
- [ ] **Step 5:** Commit `feat: convert granules into range-routed staging fragments`.

---

### Task 6: Compact — sorted GeoParquet output

**Files:**
- Create: `processor/src/atl06_processor/compact.py`, `tests/processor/test_compact.py`

**Interfaces:**
- Consumes: `RunPaths`, `SCHEMA`, `geo_metadata`.
- Produces:
  - `FileStats` dataclass: `range_id, path, rows, bytes, row_groups, hash_lo, hash_hi, time_min_ns, time_max_ns, xmin, ymin, xmax, ymax`; JSON round-trip.
  - `compact_range(fs, paths: RunPaths, range_id: int, row_group_size: int) -> FileStats | None` — lists `paths.fragment_dir(range_id)`; returns `None` (and writes nothing) if empty; else reads all fragments, `concat_tables`, `sort_by("stac_hash")`, attaches `geo` metadata (bbox from actual data) and writes `paths.data_file(range_id)` with `compression="zstd"`, `write_statistics=True`, `row_group_size=row_group_size`; writes stats JSON to `paths.compact_stats(range_id)`.

- [ ] **Step 1: Write failing tests:**

```python
def test_compact_sorts_by_hash(...):      # pq.read_table(...)["stac_hash"] is non-decreasing
def test_compact_row_groups(...):         # 2_500 rows, row_group_size=1000 → 3 row groups
def test_compact_geo_metadata(...):       # json.loads(md[b"geo"])["columns"]["geometry"]["covering"] present
def test_stats_exclude_fill_values(...):  # row-group max of h_li < 1e30 when inputs contained fills
def test_compact_range_with_no_fragments_returns_none(...)
def test_bbox_stats_present(...):         # metadata.row_group(0).column("bbox.xmin").statistics.has_min_max
```

- [ ] **Step 2:** Run → FAIL. **Step 3:** Implement. **Step 4:** Run → PASS.
- [ ] **Step 5:** Commit `feat: compact range fragments into sorted GeoParquet`.

---

### Task 7: Finalize + local end-to-end pipeline test

**Files:**
- Create: `processor/src/atl06_processor/finalize.py`, `tests/processor/test_end_to_end.py`

**Interfaces:**
- Consumes: everything above.
- Produces:
  - `finalize_run(fs, paths: RunPaths) -> dict` — reads all `stats/convert/*.json` and `stats/compact/*.json`; writes `manifest/dataset.parquet` (one row per `FileStats`) and `manifest/report.json` with: `granules`, `rows_in`, `rows_out_convert`, `rows_out_compact`, `dropped_out_of_year`, `files`, `rows_per_file {min, median, max, max_over_median}`, `bytes_per_file {min, median, max}`, `fanout {mean, p50, p95, max}`; raises `ValueError` if `rows_out_convert != rows_out_compact`.

- [ ] **Step 1: Write failing test** `test_pipeline_end_to_end`: build ~20 synthetic granules spread over 2023 (plus one straddling Dec 31) with clustered high-latitude points, then call `sample_granule` on all → `SplitTable.from_sample(target_rows_per_file=rows/5)` → `convert_granule` each → `compact_range` each → `finalize_run`. Assert: report rows balance; `files` ≥ 4; `max_over_median < 2`; every data file sorted; duckdb `SELECT count(*) FROM read_parquet('<root>/data/*.parquet')` == `rows_out_compact`.
- [ ] **Step 2:** Run → FAIL. **Step 3:** Implement `finalize.py`. **Step 4:** Run → PASS.
- [ ] **Step 5:** Commit `feat: finalize report and local end-to-end test`.

---

### Task 8: Inventory, NSIDC credentials, Lambda handlers, Docker image

**Files:**
- Create: `processor/src/atl06_processor/{inventory,credentials,handlers}.py`, `processor/Dockerfile`, `tests/processor/test_inventory.py`, `tests/processor/test_handlers.py`

**Interfaces:**
- Consumes: all processor modules; `Settings`.
- Produces:
  - `build_inventory(granules: list[tuple[str, str]], sample_fraction: float, seed: int, limit: int | None) -> tuple[pa.Table, list[dict], list[dict]]` — input `(granule_name, s3_url)`, sorted by name, optional `limit` (spike); returns granule table `(srcid int32, granule string, s3_url string)`, all items `[{srcid, granule, s3_url}]`, and the sampled subset (at least 1 item; all items if `sample_fraction >= 1`).
  - `search_granules(settings, start: datetime, end: datetime) -> list[tuple[str, str]]` — `earthaccess.search_data(short_name=..., version=..., temporal=(start, end))`; take each result's direct-access `s3://nsidc-cumulus-prod-protected/...h5` link. Untested beyond a mock (needs network); README notes it.
  - `NsidcS3(secret_name: str)` — `.fs() -> s3fs.S3FileSystem`; fetches EDL creds from Secrets Manager (JSON `{"username","password"}`), `earthaccess.login(strategy="environment")`, `earthaccess.get_s3_credentials(daac="NSIDC")`; caches per container and refreshes when older than 50 minutes (injectable clock for tests).
  - Handlers (all read `Settings()` from env; event `run_prefix` overrides settings):
    - `inventory_handler(event)` — event `{start?, end?, limit?}` (default: settings year) → writes granule table, `granules.json`, `sample.json`; returns `{"granules": n, "sampled": m, "run_prefix": ...}`.
    - `sample_handler(event)` — Distributed Map batch `{"Items": [{srcid, granule, s3_url}, ...]}` → writes `sample/{stem}.parquet` (`stac_hash` column + `row_count` in schema metadata) per item.
    - `plan_splits_handler(event)` → reads all sample parts, computes `SplitTable`, saves `splits.parquet` and `ranges.json` (`[{range_id}]`); returns `{"ranges": n, "est_total_rows": ...}`.
    - `convert_handler(event)` — batch of items → `convert_granule` each; returns list of `{granule, rows_out, ranges_touched}`.
    - `compact_handler(event)` — batch of `{range_id}` → `compact_range` each.
    - `finalize_handler(event)` → `finalize_run`; returns report.
  - `processor/Dockerfile`: `FROM public.ecr.aws/lambda/python:3.12`, install the package with `uv pip install --system`, default `CMD ["atl06_processor.handlers.convert_handler"]` (CDK overrides per function).

- [ ] **Step 1: Write failing tests:**

```python
def test_build_inventory_assigns_srcid_sorted(): ...      # srcid follows sorted granule name
def test_build_inventory_limit_and_full_sample(): ...      # limit=3, sample_fraction=1.0 → 3 items, 3 sampled
def test_nsidc_creds_refresh_after_50_min(monkeypatch): ... # fake clock + stubbed earthaccess → 2 fetches over 51 min, 1 within
def test_handlers_end_to_end_local(monkeypatch, make_granule, tmp_path): ...
    # monkeypatch handler fs factories to local fs & src fs to local paths;
    # inventory(from injected list) → sample → plan_splits → convert → compact → finalize; report balances
```
Handlers obtain filesystems via module-level `get_output_fs()` / `get_source_fs()` functions so tests can monkeypatch them.

- [ ] **Step 2:** Run → FAIL. **Step 3:** Implement. **Step 4:** Run → PASS.
- [ ] **Step 5:** `docker build -f processor/Dockerfile .` succeeds (skip if Docker is unavailable; note it).
- [ ] **Step 6:** Commit `feat: inventory, NSIDC credentials, Lambda handlers, image`.

---

### Task 9: CDK stack (Lambda + Step Functions)

**Files:**
- Create: `cdk/pyproject.toml`, `cdk/cdk.json`, `cdk/app.py`, `cdk/atl06_cdk/{__init__,settings,stack}.py`, `tests/cdk/test_stack.py`

**Interfaces:**
- Consumes: `processor/Dockerfile`, handler names (Task 8), `Settings` env names.
- Produces: `Atl06StacHashStack(scope, id, settings: StackSettings)`; `StackSettings` (env prefix `CDK_`): `account`, `region = "us-west-2"`, `bucket_name: str | None` (None → stack creates one, `RETAIN`), `edl_secret_name = "atl06/earthdata"`, `convert_concurrency = 50`, `compact_concurrency = 100`, `convert_batch_size = 1`, `convert_memory_mb = 4096`, `compact_memory_mb = 10240`.
  - One `DockerImageAsset`-backed `DockerImageFunction` per handler (cmd override), timeout 15 min, `ephemeral_storage` 10 GB for compact, env from `Settings` names, read on the secret, read/write on the bucket.
  - State machine: `Inventory` (LambdaInvoke) → `Sample` (DistributedMap, `ItemReader` S3 JSON `manifest/sample.json`, `ItemBatcher` max 5, concurrency `convert_concurrency`) → `PlanSplits` → `Convert` (DistributedMap over `manifest/granules.json`, batcher `convert_batch_size`, `ResultWriter` to `results/convert/`, `toleratedFailurePercentage` 0) → `Compact` (DistributedMap over `manifest/ranges.json`, batcher 1) → `Finalize`. Execution input `{run_prefix, start?, end?, limit?}` is passed to every step. Retries: 3 attempts, backoff 2.0, on `Lambda.TooManyRequestsException`, `States.TaskFailed`.

- [ ] **Step 1: Write failing tests** with `aws_cdk.assertions.Template`:

```python
def test_six_image_lambdas(): assert len(template.find_resources("AWS::Lambda::Function", {"Properties": {"PackageType": "Image"}})) == 6  # inventory, sample, plan_splits, convert, compact, finalize
def test_state_machine_has_three_distributed_maps(): # definition string contains 3 '"Mode":"DISTRIBUTED"'
def test_convert_concurrency_from_settings(): # MaxConcurrency == 50
def test_compact_has_10gb_memory_and_storage(): # MemorySize == 10240, EphemeralStorage.Size == 10240
```

- [ ] **Step 2:** Run `uv run pytest tests/cdk -v` → FAIL. **Step 3:** Implement. **Step 4:** Run → PASS; `cd cdk && uv run cdk synth` succeeds.
- [ ] **Step 5:** Commit `feat: CDK stack with Step Functions pipeline`.

---

### Task 10: Benchmark harness

**Files:**
- Create: `benchmark/pyproject.toml` (deps `duckdb>=2.0`, `pyarrow`), `benchmark/queries.toml`, `benchmark/src/atl06_benchmark/{__init__,config,sql,runner,cli}.py`, `tests/benchmark/test_sql.py`, `tests/benchmark/test_runner.py`

**Interfaces:**
- Produces:
  - `Query(name: str, bbox: tuple[float,float,float,float] | None, start: str | None, end: str | None, columns: list[str])`, `Dataset(name: str, uri: str)` (glob, e.g. `s3://bucket/runs/x/data/*.parquet`), `Suite(datasets, queries, repeats: int = 3)`; `load_suite(path) -> Suite`.
  - `build_sql(dataset_uri: str, q: Query) -> str` — `SELECT <columns> FROM read_parquet('<uri>') WHERE …` using only: `bbox.xmin >= minx AND bbox.xmax <= maxx AND bbox.ymin >= miny AND bbox.ymax <= maxy` and `time_ns >= TIMESTAMP '<start>' AND time_ns < TIMESTAMP '<end>'`. Never mentions `stac_hash`.
  - `QueryMetrics(dataset, query, repeat, files_total, files_read, row_groups_total, row_groups_scanned, bytes_read, rows, wall_s)`.
  - `run_query(con, sql, files_total) -> QueryMetrics` — enables `PRAGMA enable_profiling='json'` with `profiling_output` to a temp file and `SET profiling_mode='detailed'`; fetches result as Arrow; parses (verified against DuckDB 2.0 output): `io.total_bytes_read`, `query.total_row_groups_scanned`, `query.total_row_groups_to_scan`, and `"Total Files Read"` from the `TABLE_SCAN` operator's `extra_info`. Disables DuckDB's external file cache (`SET enable_external_file_cache=false`) so repeats are cold reads.
  - `run_suite(suite) -> pa.Table` — one row per (dataset, query, repeat); S3 auth via `CREATE SECRET (TYPE s3, PROVIDER credential_chain)`.
  - CLI: `atl06-bench run --suite benchmark/queries.toml --out results.parquet`.
  - `queries.toml` default suite (spec §10): `basin_full_year` (Jakobshavn ≈ `[-50.5, 68.8, -48.5, 69.4]`, 2023), `ice_sheet_one_day_greenland` (`[-74, 59, -10, 84]`, 2023-07-01..07-02), `ice_sheet_one_week_antarctica` (`[-180, -90, 180, -60]`, 2023-03-01..03-08), `basin_one_week` (Jakobshavn, 2023-07-01..07-08). Columns: `time_ns, h_li, geometry`.

- [ ] **Step 1: Write failing tests:**

```python
def test_build_sql_uses_only_plain_predicates():
    sql = build_sql("x/*.parquet", Query("q", (-1, -2, 3, 4), "2023-01-01", "2023-02-01", ["h_li"]))
    assert "stac_hash" not in sql and "bbox.xmin >= -1" in sql and "time_ns < TIMESTAMP '2023-02-01'" in sql

def test_run_query_reports_pruning(tmp_path):
    # write 1 file, 10 row groups sorted on bbox.xmin; query a narrow xmin range
    m = run_query(duckdb.connect(), sql, files_total=1)
    assert m.row_groups_total == 10 and m.row_groups_scanned <= 2 and m.bytes_read > 0 and m.files_read == 1

def test_load_suite_default(): assert {q.name for q in load_suite("benchmark/queries.toml").queries} >= {"basin_full_year", "ice_sheet_one_day_greenland"}
```

- [ ] **Step 2:** Run → FAIL. **Step 3:** Implement. **Step 4:** Run → PASS.
- [ ] **Step 5:** Commit `feat: DuckDB benchmark harness with pushdown metrics`.

---

### Task 11: Naive baseline dataset

**Files:**
- Create: `benchmark/src/atl06_benchmark/baseline.py`, `tests/benchmark/test_baseline.py`; Modify: `benchmark/src/atl06_benchmark/cli.py`

**Interfaces:**
- Produces: `build_baseline(con, src_uri: str, dst_uri: str, row_group_size: int) -> int` — DuckDB `COPY (SELECT *, month(time_ns) AS month FROM read_parquet(src) ORDER BY srcid, time_ns) TO dst (FORMAT parquet, PARTITION_BY (month), ROW_GROUP_SIZE n, COMPRESSION zstd)`; returns rows written. Represents "granule/time order, partitioned by month". CLI `atl06-bench baseline --src … --dst … --row-group-size …`.

- [ ] **Step 1: Write failing test** `test_baseline_roundtrip_rowcount` (local files; count equal, ≥1 month partition dir).
- [ ] **Step 2:** Run → FAIL. **Step 3:** Implement. **Step 4:** Run → PASS.
- [ ] **Step 5:** Commit `feat: month-partitioned baseline builder`.

---

### Task 12: README, deploy, and spike run

**Files:**
- Create: `README.md`, `docs/spike-results.md`

- [ ] **Step 1:** Write `README.md`: setup (`uv sync`), tests, creating the EDL secret (`aws secretsmanager create-secret --name atl06/earthdata --secret-string '{"username":"…","password":"…"}' --region us-west-2`), deploy (`cd cdk && uv run cdk deploy`), starting a spike execution (`aws stepfunctions start-execution --input '{"run_prefix":"runs/spike-1","limit":40}'`), running the benchmark; an **Assumptions** list (SlideRule mappings, leap-second constant, fill→null, sampling defaults, target_rows_per_file estimate).
- [ ] **Step 2: STOP and ask the user** for the AWS account/profile to deploy into and for the Earthdata credentials (or confirmation that they created the secret). Do not proceed without them.
- [ ] **Step 3:** Deploy; run a spike execution with `limit` ≈ 40 granules over 2023, `sample_granule_fraction = 1.0`, and `target_rows_per_file` lowered so the spike yields ≥ 10 files (e.g. total rows / 20).
- [ ] **Step 4:** Validate against real data: spot-check one granule's converted rows vs `h5py` reads for `h_li`, `time_ns`, `gt`, `spot`; confirm `report.json` row balance.
- [ ] **Step 5:** Build the baseline from the spike output and run `atl06-bench run` (from an EC2 instance or CloudShell in us-west-2) against both.
- [ ] **Step 6:** Record in `docs/spike-results.md`: report.json highlights (fan-out mean/p95/max, rows/bytes per file spread, dropped rows), Lambda max memory and duration per step (CloudWatch), estimated bytes/row → recommended `target_rows_per_file` for 512 MB–1 GB, benchmark table for both datasets, and any surprises.
- [ ] **Step 7:** Commit `docs: README and spike results`.
