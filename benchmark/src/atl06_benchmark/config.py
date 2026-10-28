"""Benchmark suite definition, loaded from TOML."""

import tomllib
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Query:
    name: str
    bbox: tuple[float, float, float, float] | None
    start: str | None
    end: str | None
    columns: list[str] = field(default_factory=lambda: ["*"])


@dataclass(frozen=True)
class Dataset:
    name: str
    uri: str


@dataclass
class Suite:
    datasets: list[Dataset]
    queries: list[Query]
    repeats: int = 3


def load_suite(path: str | Path) -> Suite:
    raw = tomllib.loads(Path(path).read_text())
    queries = [
        Query(
            name=q["name"],
            bbox=tuple(q["bbox"]) if "bbox" in q else None,
            start=q.get("start"),
            end=q.get("end"),
            columns=q.get("columns", ["*"]),
        )
        for q in raw["queries"]
    ]
    datasets = [Dataset(d["name"], d["uri"]) for d in raw["datasets"]]
    return Suite(datasets, queries, raw.get("repeats", 3))
