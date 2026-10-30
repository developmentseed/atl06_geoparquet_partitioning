"""Every object key a pipeline run reads or writes, relative to the run root."""

from dataclasses import dataclass


def _stem(granule: str) -> str:
    return granule.rsplit("/", 1)[-1].removesuffix(".h5")


@dataclass(frozen=True)
class RunPaths:
    root: str

    def _p(self, key: str) -> str:
        return f"{self.root.rstrip('/')}/{key}"

    def granules_table(self) -> str:
        return self._p("manifest/granules.parquet")

    def granules_json(self) -> str:
        return self._p("manifest/granules.json")

    def sample_json(self) -> str:
        return self._p("manifest/sample.json")

    def sample_dir(self) -> str:
        return self._p("sample")

    def sample_part(self, granule: str) -> str:
        return self._p(f"sample/{_stem(granule)}.parquet")

    def splits(self) -> str:
        return self._p("manifest/splits.parquet")

    def ranges_json(self) -> str:
        return self._p("manifest/ranges.json")

    def fragment_dir(self, range_id: int) -> str:
        return self._p(f"staging/range={range_id:05d}")

    def fragment(self, range_id: int, granule: str) -> str:
        return f"{self.fragment_dir(range_id)}/{_stem(granule)}.parquet"

    def convert_stats_dir(self) -> str:
        return self._p("stats/convert")

    def convert_stats(self, granule: str) -> str:
        return f"{self.convert_stats_dir()}/{_stem(granule)}.json"

    def data_dir(self) -> str:
        return self._p("data")

    def data_file(self, range_id: int, part: int) -> str:
        return self._p(f"data/part-{range_id:05d}-{part:02d}.parquet")

    def compact_stats_dir(self) -> str:
        return self._p("stats/compact")

    def compact_stats(self, range_id: int, part: int) -> str:
        return f"{self.compact_stats_dir()}/{range_id:05d}-{part:02d}.json"

    def dataset_manifest(self) -> str:
        return self._p("manifest/dataset.parquet")

    def iceberg_dir(self) -> str:
        return self._p("iceberg")

    def report(self) -> str:
        return self._p("manifest/report.json")
