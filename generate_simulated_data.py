"""Command-line entry point for reproducible simulated shipment data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.simulation import SimulationSettings, save_simulated_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate labelled simulated shipment history.")
    parser.add_argument("--rows", type=int, default=12_000)
    parser.add_argument("--start-date", default="2025-01-01")
    parser.add_argument("--end-date", default="2026-09-30")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="data/simulated_shipments.csv")
    parser.add_argument("--dictionary", default="data/data_dictionary.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = SimulationSettings(
        rows=args.rows,
        start_date=args.start_date,
        end_date=args.end_date,
        seed=args.seed,
    )
    df = save_simulated_data(args.output, args.dictionary, settings)
    metadata = {
        "data_source": "SIMULATED",
        "rows": len(df),
        "start_date": str(df["ship_date"].min()),
        "end_date": str(df["ship_date"].max()),
        "seed": args.seed,
        "delay_rate": round(float(df["is_delayed"].mean()), 4),
        "warning": "Synthetic data for product and pipeline development; not evidence of real-world model performance.",
    }
    metadata_path = Path(args.output).with_name("simulation_metadata.json")
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()

