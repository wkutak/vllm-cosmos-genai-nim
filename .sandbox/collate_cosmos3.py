#!/usr/bin/env python
"""Collate `vllm bench serve` JSON results into a single perf-tradeoff CSV.

Reads <result-dir>/<label>.json for each label and emits one row per
checkpoint with throughput, TTFT, TPOT/ITL, and end-to-end latency stats.
"""

import argparse
import csv
import json
from pathlib import Path

# (csv_column, json_key). Missing keys are written as empty cells.
COLUMNS = [
    ("label", "label"),
    ("checkpoint", "checkpoint"),
    ("num_prompts", "completed"),
    ("duration_s", "duration"),
    # Throughput
    ("req_throughput_per_s", "request_throughput"),
    ("output_tok_per_s", "output_throughput"),
    ("total_tok_per_s", "total_token_throughput"),
    # Time to first token (ms)
    ("ttft_mean_ms", "mean_ttft_ms"),
    ("ttft_median_ms", "median_ttft_ms"),
    ("ttft_p99_ms", "p99_ttft_ms"),
    # Time per output token (ms)
    ("tpot_mean_ms", "mean_tpot_ms"),
    ("tpot_median_ms", "median_tpot_ms"),
    ("tpot_p99_ms", "p99_tpot_ms"),
    # Inter-token latency (ms)
    ("itl_mean_ms", "mean_itl_ms"),
    ("itl_median_ms", "median_itl_ms"),
    # End-to-end latency percentiles (ms)
    ("e2e_mean_ms", "mean_e2el_ms"),
    ("e2e_median_ms", "median_e2el_ms"),
    ("e2e_p90_ms", "p90_e2el_ms"),
    ("e2e_p99_ms", "p99_e2el_ms"),
]


def fmt(value):
    if isinstance(value, float):
        return f"{value:.4g}"
    return value


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--result-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--labels", required=True, help="comma-separated labels")
    args = ap.parse_args()

    result_dir = Path(args.result_dir)
    rows = []
    for label in args.labels.split(","):
        label = label.strip()
        path = result_dir / f"{label}.json"
        if not path.exists():
            print(f"!! missing result for '{label}': {path} (skipping)")
            continue
        data = json.loads(path.read_text())
        data.setdefault("label", label)
        rows.append({col: fmt(data.get(key, "")) for col, key in COLUMNS})

    if not rows:
        raise SystemExit("no results found to collate")

    out = Path(args.out)
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[c for c, _ in COLUMNS])
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {out}")


if __name__ == "__main__":
    main()
