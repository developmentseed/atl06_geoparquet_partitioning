"""Register a run's output files as an Iceberg v2 table, without rewriting them.

Query engines that read Iceberg (DuckDB ``iceberg_scan``) prune whole files from the
manifests' per-file column bounds, so they open only the footers of files that can match.
"""

import tempfile

from pyiceberg.catalog.sql import SqlCatalog
from pyiceberg.table import _parquet_files_to_data_files

from atl06_processor.paths import RunPaths
from atl06_processor.schema import OUTPUT_SCHEMA

NAMESPACE = "atl06"


# Position of ``nan_value_counts`` in pyiceberg's DataFile record (read-only property).
_NAN_VALUE_COUNTS = 9


def register_table(
    root_uri: str,
    paths: RunPaths,
    data_keys: list[str],
    nan_counts: dict[str, dict[str, int]] | None = None,
    properties: dict | None = None,
) -> str:
    """Creates the table under ``<root_uri>/<run>/iceberg``; returns its metadata.json URI.

    ``nan_counts`` maps data key → {column: NaN count}. Parquet statistics carry no NaN
    counts, and without them engines cannot prune float columns on ``>=``/``>`` (NaN sorts
    above every number in DuckDB), so counted columns are written into the manifest.
    The SQLite catalog is throwaway: readers open the metadata.json directly.
    """
    location = f"{root_uri.rstrip('/')}/{paths.iceberg_dir()}"
    with tempfile.TemporaryDirectory() as tmp:
        catalog = SqlCatalog(
            NAMESPACE, uri=f"sqlite:///{tmp}/catalog.db", warehouse=location, **(properties or {})
        )
        catalog.create_namespace(NAMESPACE)
        table = catalog.create_table(
            f"{NAMESPACE}.dataset",
            schema=OUTPUT_SCHEMA,
            location=location,
            properties={"format-version": "2"},
        )
        root = root_uri.rstrip("/")
        field_ids = {f.name: f.field_id for f in table.schema().fields}
        with table.transaction() as tx:
            # add_files would set this; the files carry no Iceberg field ids.
            tx.set_properties(
                **{"schema.name-mapping.default": table.schema().name_mapping.model_dump_json()}
            )
            data_files = _parquet_files_to_data_files(
                tx.table_metadata, [f"{root}/{k}" for k in data_keys], table.io
            )
            with tx.update_snapshot().fast_append() as append:
                for key, data_file in zip(data_keys, data_files, strict=True):
                    counts = (nan_counts or {}).get(key, {})
                    data_file[_NAN_VALUE_COUNTS] = {field_ids[c]: n for c, n in counts.items()}
                    append.append_data_file(data_file)
        return catalog.load_table(f"{NAMESPACE}.dataset").metadata_location
