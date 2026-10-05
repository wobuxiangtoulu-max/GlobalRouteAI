"""Command-line model training and evaluation."""

from __future__ import annotations

import argparse
import json

import pandas as pd

from src.modeling import MODEL_DISPLAY_NAMES, train_all_models


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train delay prediction models with a chronological holdout.")
    parser.add_argument("--data", default="data/simulated_shipments.csv")
    parser.add_argument("--output-dir", default="models")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=list(MODEL_DISPLAY_NAMES),
        default=list(MODEL_DISPLAY_NAMES),
    )
    parser.add_argument("--test-fraction", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--require-catboost", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.data)
    metrics, manifest = train_all_models(
        df,
        output_dir=args.output_dir,
        model_names=args.models,
        test_fraction=args.test_fraction,
        random_state=args.seed,
        require_catboost=args.require_catboost,
    )
    print(metrics[["display_name", "roc_auc", "pr_auc", "precision", "recall", "f1", "brier_score"]].to_string(index=False))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

