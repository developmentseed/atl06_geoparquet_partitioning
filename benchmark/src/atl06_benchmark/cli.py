"""``atl06-bench`` command line."""

import argparse

import pyarrow.parquet as pq

from atl06_benchmark.baseline import build_baseline
from atl06_benchmark.config import Dataset, Suite, load_suite
from atl06_benchmark.runner import connect, run_suite


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="atl06-bench")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="run a query suite and write metrics")
    run.add_argument("--suite", default="benchmark/queries.toml")
    run.add_argument("--out", default="results.parquet")
    base = sub.add_parser("baseline", help="build the month-partitioned baseline dataset")
    base.add_argument("--src", required=True, help="source glob, e.g. s3://b/runs/x/data/*.parquet")
    base.add_argument("--dst", required=True, help="destination directory")
    base.add_argument("--row-group-size", type=int, default=500_000)
    args = parser.parse_args(argv)

    if args.command == "baseline":
        con = connect(Suite([Dataset("src", args.src), Dataset("dst", args.dst)], []))
        print(build_baseline(con, args.src, args.dst, args.row_group_size), "rows written")

    if args.command == "run":
        table = run_suite(load_suite(args.suite))
        pq.write_table(table, args.out)
        print(table)
