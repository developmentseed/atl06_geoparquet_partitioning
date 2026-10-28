import fsspec
import numpy as np
from atl06_processor.splits import SplitTable


def test_from_sample_balanced():
    h = np.random.default_rng(0).integers(0, 2**63, 100_000, dtype=np.uint64)
    st = SplitTable.from_sample(h, est_total_rows=1_000_000, target_rows_per_file=100_000)
    assert st.n_ranges == 10
    counts = np.bincount(st.range_ids(h))
    assert counts.max() / counts.min() < 1.1


def test_skewed_sample_still_balanced():
    rng = np.random.default_rng(1)
    dense = rng.integers(2**40, 2**40 + 10**6, 90_000, dtype=np.uint64)
    sparse = rng.integers(0, 2**63, 10_000, dtype=np.uint64)
    h = np.concatenate([dense, sparse])
    st = SplitTable.from_sample(h, est_total_rows=100_000, target_rows_per_file=10_000)
    counts = np.bincount(st.range_ids(h))
    assert counts.max() / counts.min() < 1.2


def test_duplicate_cuts_deduped():
    st = SplitTable.from_sample(np.full(1000, 42, np.uint64), 1000, 100)
    assert np.all(np.diff(st.cuts.astype(np.int64)) > 0)


def test_value_on_cut_goes_right():
    st = SplitTable(np.array([100, 200], np.uint64))
    assert list(st.range_ids(np.array([99, 100, 199, 200], np.uint64))) == [0, 1, 1, 2]


def test_ranges_cover_hash_space():
    st = SplitTable(np.array([100, 200], np.uint64))
    assert st.ranges() == [(0, 0, 100), (1, 100, 200), (2, 200, 2**63)]


def test_save_load_roundtrip(tmp_path):
    fs = fsspec.filesystem("file")
    st = SplitTable(np.array([5, 9, 2**62], np.uint64))
    st.save(fs, str(tmp_path / "splits.parquet"))
    assert np.array_equal(SplitTable.load(fs, str(tmp_path / "splits.parquet")).cuts, st.cuts)
