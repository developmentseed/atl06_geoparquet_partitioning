"""Sample one granule: in-year row count plus a random subsample of its hashes."""

import io

import h5py
import numpy as np

from atl06_processor import storage
from atl06_processor.granule import GranuleInfo
from atl06_processor.hashing import filter_and_hash, make_hasher
from atl06_processor.reader import iter_beam_batches


def sample_granule(
    src_store, src_key: str, info: GranuleInfo, year: int, row_fraction: float, seed: int
) -> tuple[int, np.ndarray]:
    hasher = make_hasher(year)
    rng = np.random.default_rng(seed)
    count, parts = 0, []
    with h5py.File(io.BytesIO(storage.read_bytes(src_store, src_key)), "r") as h5:
        for batch in iter_beam_batches(h5, info, srcid=0):
            hashed, _, _ = filter_and_hash(batch, year, hasher)
            count += len(hashed)
            h = hashed["stac_hash"].to_numpy(zero_copy_only=False)
            parts.append(h[rng.random(len(h)) < row_fraction])
    hashes = np.concatenate(parts) if parts else np.array([], np.uint64)
    return count, hashes.astype(np.uint64)
