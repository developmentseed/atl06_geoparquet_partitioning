"""Object storage helpers over obstore. Keys are store-relative, with no leading slash."""

import asyncio
import json
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


def put_bytes(store: ObjectStore, key: str, data: bytes) -> None:
    obstore.put(store, key, data)


def put_parquet(store: ObjectStore, key: str, table: pa.Table, **kwargs) -> None:
    sink = pa.BufferOutputStream()
    pq.write_table(table, sink, **kwargs)
    # obstore.put rejects pyarrow Buffers, and pyarrow can't write into obstore's writer
    # (its `closed` is a method), so upload a bytes copy; outputs are <= ~1 GB.
    obstore.put(store, key, sink.getvalue().to_pybytes())


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
