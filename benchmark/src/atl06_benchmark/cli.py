"""``atl06-bench`` command line."""

import argparse

import pyarrow.parquet as pq

from atl06_benchmark.config import load_suite
from atl06_benchmark.runner import run_suite


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="atl06-bench")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="run a query suite and write metrics")
    run.add_argument("--suite", default="benchmark/queries.toml")
    run.add_argument("--out", default="results.parquet")
    args = parser.parse_args(argv)

    if args.command == "run":
        table = run_suite(load_suite(args.suite))
        pq.write_table(table, args.out)
        print(table)
