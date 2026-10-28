"""Balanced contiguous stac-hash ranges, one per output file."""

import math
from dataclasses import dataclass

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from atl06_processor import storage

HASH_END = 2**63  # stac-hash uses 63 bits; every hash is < 2**63


@dataclass
class SplitTable:
    """Interior cut points; range i is [lo_i, hi_i) with lo_0 = 0 and hi_last = 2**63."""

    cuts: np.ndarray

    def __post_init__(self) -> None:
        self.cuts = np.asarray(self.cuts, dtype=np.uint64)

    @property
    def n_ranges(self) -> int:
        return len(self.cuts) + 1

    @classmethod
    def from_sample(
        cls, hashes: np.ndarray, est_total_rows: int, target_rows_per_file: int
    ) -> "SplitTable":
        n = max(1, math.ceil(est_total_rows / target_rows_per_file))
        if n == 1 or len(hashes) == 0:
            return cls(np.array([], np.uint64))
        qs = [k / n for k in range(1, n)]
        cuts = np.quantile(np.asarray(hashes, np.uint64), qs, method="inverted_cdf")
        cuts = np.unique(cuts.astype(np.uint64))
        return cls(cuts[cuts > 0])

    def range_ids(self, hashes: np.ndarray) -> np.ndarray:
        return np.searchsorted(self.cuts, np.asarray(hashes, np.uint64), side="right").astype(
            np.int32
        )

    def ranges(self) -> list[tuple[int, int, int]]:
        bounds = [0, *(int(c) for c in self.cuts), HASH_END]
        return [(i, bounds[i], bounds[i + 1]) for i in range(self.n_ranges)]

    def save(self, store, key: str) -> None:
        rows = self.ranges()
        table = pa.table(
            {
                "range_id": pa.array([r[0] for r in rows], pa.int32()),
                "hash_lo": pa.array([r[1] for r in rows], pa.uint64()),
                "hash_hi": pa.array([r[2] for r in rows], pa.uint64()),
            }
        )
        storage.put_parquet(store, key, table)

    @classmethod
    def load(cls, store, key: str) -> "SplitTable":
        table = pq.read_table(pa.BufferReader(storage.read_bytes(store, key)))
        return cls(table["hash_lo"].to_numpy()[1:])
