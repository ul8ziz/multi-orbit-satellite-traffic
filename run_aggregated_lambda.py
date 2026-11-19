# -*- coding: utf-8 -*-
"""CLI entrypoint for chunked hourly lambda aggregation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent


def _ensure_src_on_path() -> None:
    r = str(PROJECT_ROOT)
    if r not in sys.path:
        sys.path.insert(0, r)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Chunked telecom traffic -> hourly lambda table.")
    p.add_argument(
        "--input",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw" / "Applications_data_sample.csv",
        help="Path to input CSV (~41M rows).",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "aggregated_lambda.csv",
        help="Output CSV path.",
    )
    p.add_argument("--alpha", type=float, default=1.0, help="Lambda scaling coefficient.")
    p.add_argument("--chunk-rows", type=int, default=500_000, help="CSV chunk size.")
    p.add_argument(
        "--max-mb-per-user-per-hour",
        type=float,
        default=1000.0,
        help="Outlier threshold for total_mb per row.",
    )
    return p.parse_args()


def main() -> int:
    _ensure_src_on_path()
    from src.integration.aggregate_lambda_chunked import build_aggregated_lambda

    args = parse_args()
    in_path = args.input if args.input.is_absolute() else (PROJECT_ROOT / args.input)
    out_path = args.output if args.output.is_absolute() else (PROJECT_ROOT / args.output)

    if not in_path.is_file():
        fallback = PROJECT_ROOT / "data" / "raw" / "traffic.csv"
        if fallback.is_file():
            in_path = fallback
        else:
            raise FileNotFoundError(
                f"Input CSV not found: {args.input}. Also fallback missing: {fallback}"
            )

    out = build_aggregated_lambda(
        in_path,
        out_path,
        alpha=args.alpha,
        chunk_rows=args.chunk_rows,
        max_mb_per_user_per_hour=args.max_mb_per_user_per_hour,
    )
    print(out.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
