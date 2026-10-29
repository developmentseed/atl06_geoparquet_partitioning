# Iceberg Table, Bounded Compact, Restart-from-Compact Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Register the output as an Iceberg v2 table DuckDB can prune from manifests, make Compact survive oversized ranges, and let an existing run restart at Compact.

**Architecture:** Compact downloads a range's fragments to ephemeral disk, counts rows from `stac_hash` alone, splits ranges above `max_rows_per_file` at hash quantiles, and writes each part sorted one row group at a time, transforming to the new output schema (`time_ns` int64, `time` µs, `longitude`/`latitude`). Finalize registers every data file in an Iceberg v2 table with pyiceberg `add_files`. A Choice state at the start of the state machine routes `"from_step": "compact"` to a PrepareCompact Lambda that clears old outputs and hands CompactMap the existing ranges. The benchmark gains an `iceberg` dataset format.

**Tech Stack:** as before, plus `pyiceberg[sql-sqlite,pyarrow]>=0.12` (processor) and DuckDB's `iceberg` extension (benchmark).

**Spec:** `docs/superpowers/specs/2026-10-28-atl06-stac-hash-geoparquet-design.md` §14 (amendment), on top of the rest of that spec.

## Global Constraints

- Staging fragments keep `SCHEMA` (unchanged); only Compact output uses `OUTPUT_SCHEMA`. An existing run's `staging/` must compact with the new code.
- `OUTPUT_SCHEMA` = `SCHEMA` with `time_ns: int64` (ns since epoch, same values), `time: timestamp[us]` (naive) inserted right after `time_ns`, `longitude, latitude: float64` inserted right before `geometry` (equal to `bbox.xmin`, `bbox.ymin`). No `timestamp[ns]` column anywhere in output files.
- Output files: `data/part-{range:05d}-{part:02d}.parquet`; each part is a contiguous `stac_hash` sub-range, rows sorted by `stac_hash`; zstd, statistics, `row_group_size` rows per group, GeoParquet `geo` metadata with `bbox` covering.
- `max_rows_per_file` default 30,000,000 (`ATL06_MAX_ROWS_PER_FILE`, passed through as `CDK_MAX_ROWS_PER_FILE`).
- Iceberg: format version 2, created by pyiceberg `SqlCatalog` (SQLite under a temp dir) with table location `<root_uri>/<prefix>/iceberg`; files registered with `add_files` (no rewrite). `report.json["iceberg_metadata"]` = metadata.json URI.
- Benchmark predicates are plain: `time` and `longitude`/`latitude` only; never `stac_hash`.

## Review Focus

1. A range with zero fragments → no files, no stats, handler reports 0 rows (as before). Test: Task 2 `test_compact_range_with_no_fragments_returns_empty`.
2. A range whose hashes have heavy duplicates at a sub-cut → no row lost or written twice across parts. Test: Task 2 `test_split_parts_cover_every_row_once`.
3. `from_step: compact` on a prefix without `manifest/splits.parquet` → fails loudly before CompactMap. Test: Task 4 `test_prepare_compact_requires_splits`.
4. Re-running Compact for a prefix → stale `data/` and `stats/compact/` objects removed first, so Finalize's counts match. Test: Task 4 `test_prepare_compact_clears_previous_outputs`.
5. Iceberg pruning actually skips files in DuckDB on `time` and `longitude/latitude`. Test: Task 3 `test_iceberg_scan_prunes_files`.

---

### Task 1: Output schema transform and download helper

**Files:**
- Modify: `processor/src/atl06_processor/schema.py`, `processor/src/atl06_processor/storage.py`
- Test: `tests/processor/test_schema.py`, `tests/processor/test_storage.py`

**Interfaces:**
- Produces: `OUTPUT_SCHEMA: pa.Schema`; `to_output(batch: pa.Table | pa.RecordBatch) -> pa.Table` (input `SCHEMA`, output `OUTPUT_SCHEMA`); `storage.download_many(store, keys: list[str], directory: Path) -> list[Path]` (concurrent, ≤ `MAX_CONCURRENT_GETS` in flight, returns local paths in `keys` order).

- [ ] **Step 1: Failing tests** — `test_output_schema_layout` (names: `time_ns` int64 at 0, `time` timestamp[us] at 1, `longitude`,`latitude` float64 right before `geometry`; no `pa.types.is_timestamp(t) and t.unit == "ns"` field); `test_to_output_values` (time_ns int64 equals source ns; `time` equals source truncated to µs; longitude/latitude equal bbox.xmin/ymin; null time_ns stays null in both); `test_download_many_writes_files` (LocalStore, 3 keys → files with same bytes, order kept).
- [ ] **Step 2: Run** `uv run pytest tests/processor/test_schema.py tests/processor/test_storage.py -q` — Expected: FAIL (ImportError).
- [ ] **Step 3: Implement.** `to_output` builds columns with `pa.compute` casts (`cast(pa.int64())`, `cast(pa.timestamp("us"), safe=False)`), `bbox.field(...)` for lon/lat. `download_many` uses `obstore.get_async` + `bytes_async` under a semaphore and writes each file as it arrives.
- [ ] **Step 4: Run** same — Expected: PASS.
- [ ] **Step 5: Commit** `feat: output schema with int64 time_ns, µs time, lon/lat; download_many`.

### Task 2: Bounded, splitting Compact

**Files:**
- Modify: `processor/src/atl06_processor/compact.py`, `paths.py`, `config.py`, `handlers.py` (`compact_handler`)
- Test: `tests/processor/test_compact.py`, `tests/processor/test_end_to_end.py` (path/column updates only)

**Interfaces:**
- Consumes: Task 1 `to_output`, `download_many`, `OUTPUT_SCHEMA`.
- Produces: `compact_range(store, paths, range_id, row_group_size, max_rows_per_file, workdir: Path | None = None) -> list[FileStats]` (empty when no fragments); `FileStats` gains `part: int` (after `range_id`); `RunPaths.data_file(range_id, part)`, `RunPaths.compact_stats(range_id, part)` → `stats/compact/{range:05d}-{part:02d}.json`; `Settings.max_rows_per_file = 30_000_000`.

- [ ] **Step 1: Failing tests** (existing fixture: 2,500 rows in range 0) — update existing tests to the list return and new paths/columns; add `test_split_parts_cover_every_row_once` (`max_rows_per_file=1000` → 3 parts; parts' row counts sum to 2,500; each part sorted; `part[k].hash_hi < part[k+1].hash_lo`; the multiset of `stac_hash` over all parts equals the fragments'); `test_single_part_when_under_limit` (one file `part-00000-00.parquet`); `test_output_files_use_output_schema` (file schema equals `OUTPUT_SCHEMA` ignoring metadata); `test_compact_range_with_no_fragments_returns_empty`.
- [ ] **Step 2: Run** `uv run pytest tests/processor/test_compact.py -q` — Expected: FAIL.
- [ ] **Step 3: Implement.** Download fragments into a `TemporaryDirectory` (or `workdir`); `hashes = sorted stac_hash` from `pq.read_table(cols=["stac_hash"])` over the files; `n_parts = ceil(n / max_rows_per_file)`; sub-cuts = `np.unique(hashes[[k*n//n_parts for k in 1..n_parts-1]])`, giving half-open value ranges (duplicates stay in one part). Per part: `pyarrow.dataset.dataset(files, schema=SCHEMA).to_table(filter=lo <= stac_hash < hi)`, `idx = pc.sort_indices(table["stac_hash"])`, then a `pq.ParquetWriter` over `storage.open_writer`-backed sink (reuse `_ObstoreSink`; add `storage.parquet_writer(store, key, schema, **kw)` context manager) writing `to_output(table.take(idx[i:i+row_group_size]))` per row group. Extent/time stats accumulate per row group. `compact_handler` passes `settings.max_rows_per_file` and returns `{"range_id", "rows": sum}`.
- [ ] **Step 4: Run** `uv run pytest tests/processor -q` — Expected: PASS.
- [ ] **Step 5: Commit** `feat: compact splits oversized ranges and bounds memory`.

### Task 3: Iceberg registration in Finalize

**Files:**
- Create: `processor/src/atl06_processor/iceberg.py`
- Modify: `processor/pyproject.toml` (add `pyiceberg[sql-sqlite,pyarrow]>=0.12`), `finalize.py`, `handlers.py` (`finalize_handler`), `paths.py` (`iceberg_dir()`)
- Test: `tests/processor/test_iceberg.py`, `tests/processor/test_end_to_end.py`

**Interfaces:**
- Produces: `register_table(root_uri: str, paths: RunPaths, data_keys: list[str], properties: dict | None = None) -> str` (metadata.json URI); `finalize_run(store, paths, root_uri: str | None = None, iceberg_properties: dict | None = None) -> dict` — when `root_uri` is given and files exist, registers the table and sets `report["iceberg_metadata"]`. Handler passes `root_uri=f"s3://{settings.bucket}"`, `iceberg_properties={"s3.region": region}`.

- [ ] **Step 1: Failing tests** — `test_register_table_lists_every_file` (end-to-end run fixture; DuckDB `iceberg_metadata(uri)` returns exactly the data files); `test_iceberg_scan_prunes_files` (on the run fixture: a `time` window covering one synthetic granule's day opens fewer files than `files_total` per profiler "Total Files Read", and `count(*)` equals the `read_parquet` count; same for a `longitude/latitude` box); `test_finalize_report_has_iceberg_metadata`.
- [ ] **Step 2: Run** `uv run pytest tests/processor/test_iceberg.py -q` — Expected: FAIL.
- [ ] **Step 3: Implement** with `SqlCatalog("atl06", uri=f"sqlite:///{tmp}/catalog.db", warehouse=..., **properties)`, `create_table(identifier, schema=<OUTPUT_SCHEMA without metadata>, location=f"{root_uri}/{paths.iceberg_dir()}", properties={"format-version": "2"})`, `table.add_files([f"{root_uri}/{k}" for k in data_keys])`, return `table.metadata_location`.
- [ ] **Step 4: Run** `uv run pytest tests/processor -q` — Expected: PASS.
- [ ] **Step 5: Commit** `feat: finalize registers the output as an Iceberg v2 table`.

### Task 4: Restart from Compact

**Files:**
- Modify: `handlers.py` (`prepare_compact_handler`), `cdk/atl06_cdk/stack.py`, `cdk/atl06_cdk/settings.py`, `.env.sample`
- Test: `tests/processor/test_handlers.py`, `tests/cdk/test_stack.py`

**Interfaces:**
- Produces: `prepare_compact_handler(event) -> {"ranges_key", "ranges"}` (same shape PlanSplits returns); raises `ValueError` mentioning `splits` if `manifest/splits.parquet` is absent; deletes `data/`, `stats/compact/`, `iceberg/` first. Stack: 7 Lambdas; a Choice state `StartAt` routes `$.from_step == "compact"` to `PrepareCompactStep` (result into `$.plan`) → CompactMap, else Inventory…; `StackSettings.max_rows_per_file` → `ATL06_MAX_ROWS_PER_FILE`.

- [ ] **Step 1: Failing tests** — `test_prepare_compact_requires_splits`; `test_prepare_compact_clears_previous_outputs` (after a local full run, prepare then compact then finalize → same report rows, no stale keys); `test_seven_image_lambdas`; `test_from_step_choice_routes_to_prepare_compact` (definition has a Choice whose rule references `$.from_step`); `test_max_rows_per_file_passed_to_lambdas`.
- [ ] **Step 2: Run** `uv run pytest tests/processor/test_handlers.py tests/cdk -q` — Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** `uv run pytest -q` — Expected: PASS.
- [ ] **Step 5: Commit** `feat: restart a run at Compact with from_step`.

### Task 5: Benchmark on Iceberg and new columns

**Files:**
- Modify: `benchmark/src/atl06_benchmark/{config,sql,runner,baseline}.py`, `benchmark/queries.toml`
- Test: `tests/benchmark/test_sql.py`, `test_runner.py`, `test_baseline.py`

**Interfaces:**
- Produces: `Dataset(name, uri, format="parquet")`; `source_sql(ds) -> str` (`read_parquet('uri')` or `iceberg_scan('uri')`); `data_files_sql(ds)` (glob or `iceberg_metadata` file paths); `build_sql(ds, q)`; `build_files_scanned_sql(ds, q)`; predicates `time >= TIMESTAMP`, `time <`, `longitude BETWEEN minx AND maxx`, `latitude BETWEEN miny AND maxy`; `connect` loads `iceberg` when any dataset is Iceberg; baseline partitions by `month(time)`.

- [ ] **Step 1: Failing tests** — update SQL expectations to the new columns; `test_build_sql_iceberg_source`; `test_run_suite_on_iceberg_dataset` (local Iceberg table from Task 3 helper; `footers_read < files_total` for a narrow time query); `test_load_suite_format_defaults_to_parquet`.
- [ ] **Step 2: Run** `uv run pytest tests/benchmark -q` — Expected: FAIL.
- [ ] **Step 3: Implement**; update `queries.toml` with an `iceberg` dataset entry (placeholder URI) and a time-only one-day query.
- [ ] **Step 4: Run** `uv run pytest -q && uv run ruff check .` — Expected: PASS, clean.
- [ ] **Step 5: Commit** `feat: benchmark Iceberg datasets and time/lon/lat predicates`.

### Task 6: Docs and handoff

- [ ] Update `README.md` (schema change, Iceberg, `from_step`, `CDK_MAX_ROWS_PER_FILE`, benchmark format) and `.env.sample`; verify `uv run --env-file .env.sample cdk synth` succeeds (or `cdk synth` via the test template if Docker is unavailable); commit `docs: iceberg, bounded compact, restart from compact`.
