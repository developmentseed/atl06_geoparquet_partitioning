"""Object storage helpers over obstore. Keys are store-relative, with no leading slash."""

import asyncio
import json
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import urlparse

import obstore
import pyarrow as pa
import pyarrow.parquet as pq
from obstore.store import ObjectStore

MAX_CONCURRENT_GETS = 64


def split_s3_url(url: str) -> tuple[str, str]:
    parsed = urlparse(url)
    if parsed.scheme != "s3" or not parsed.netloc:
        raise ValueError(f"not an s3:// URL: {url!r}")
    return parsed.netloc, parsed.path.lstrip("/")


def read_bytes(store: ObjectStore, key: str) -> obstore.Bytes:
    """Whole object in one GET; zero-copy buffer usable by io.BytesIO and pyarrow."""
    return obstore.get(store, key).bytes()


def get_many(store: ObjectStore, keys: list[str]) -> dict[str, obstore.Bytes]:
    """Fetch many small objects concurrently (one call instead of one GET per key)."""
    if not keys:
        return {}

    async def fetch_all():
        limit = asyncio.Semaphore(MAX_CONCURRENT_GETS)

        async def fetch(key):
            async with limit:
                return key, await (await obstore.get_async(store, key)).bytes_async()

        return dict(await asyncio.gather(*(fetch(k) for k in keys)))

    return asyncio.run(fetch_all())


def download_many(store: ObjectStore, keys: list[str], directory: Path) -> list[Path]:
    """Fetch objects concurrently to local files, one per key; returns paths in ``keys`` order.

    At most MAX_CONCURRENT_GETS objects are held in memory at once.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    # Index-based names: keys from different directories may share a basename.
    files = [directory / f"{i:06d}{Path(k).suffix}" for i, k in enumerate(keys)]
    if not keys:
        return files

    async def fetch_all():
        limit = asyncio.Semaphore(MAX_CONCURRENT_GETS)

        async def fetch(key, path):
            async with limit:
                data = await (await obstore.get_async(store, key)).bytes_async()
                path.write_bytes(data)

        await asyncio.gather(*(fetch(k, f) for k, f in zip(keys, files, strict=True)))

    asyncio.run(fetch_all())
    return files


def put_bytes(store: ObjectStore, key: str, data: bytes) -> None:
    obstore.put(store, key, data)


class _ObstoreSink:
    """Writable file over obstore's multipart writer, shaped for ``pa.PythonFile``.

    pyarrow needs ``closed`` as a property and ``tell``; obstore's writer has ``closed()``
    as a method and no ``tell``.
    """

    def __init__(self, writer):
        self._writer = writer
        self._position = 0

    @property
    def closed(self) -> bool:
        return self._writer.closed()

    def writable(self) -> bool:
        return True

    def write(self, data) -> int:
        n = len(memoryview(data))
        self._writer.write(data)
        self._position += n
        return n

    def tell(self) -> int:
        return self._position

    def flush(self) -> None:
        self._writer.flush()

    def close(self) -> None:
        if not self._writer.closed():
            self._writer.close()


def put_parquet(store: ObjectStore, key: str, table: pa.Table, **kwargs) -> None:
    """Stream Parquet straight into a multipart upload; no whole-file buffer."""
    with obstore.open_writer(store, key) as writer:
        pq.write_table(table, pa.PythonFile(_ObstoreSink(writer), mode="w"), **kwargs)


@contextmanager
def parquet_writer(store: ObjectStore, key: str, schema: pa.Schema, **kwargs):
    """A ``pq.ParquetWriter`` streaming into a multipart upload, one row group at a time."""
    with obstore.open_writer(store, key) as writer:
        sink = pa.PythonFile(_ObstoreSink(writer), mode="w")
        with pq.ParquetWriter(sink, schema, **kwargs) as parquet:
            yield parquet


def put_json(store: ObjectStore, key: str, obj) -> None:
    obstore.put(store, key, json.dumps(obj).encode())


def get_json(store: ObjectStore, key: str):
    return json.loads(bytes(read_bytes(store, key)))


def size(store: ObjectStore, key: str) -> int:
    return obstore.head(store, key)["size"]


def list_keys(store: ObjectStore, prefix: str, suffix: str = "") -> list[str]:
    """Sorted keys under ``prefix`` (matched by whole path segments) ending in ``suffix``."""
    metas = obstore.list(store, prefix=prefix).collect()
    return sorted(m["path"] for m in metas if m["path"].endswith(suffix))


def prefix_exists(store: ObjectStore, prefix: str) -> bool:
    return bool(list_keys(store, prefix))


def delete_prefix(store: ObjectStore, prefix: str) -> None:
    keys = list_keys(store, prefix)
    if keys:
        obstore.delete(store, keys)
