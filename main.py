"""CLI entry point for the churn prediction ML pipeline."""

import argparse
import logging
import sys

from src.pipeline.pipeline import ChurnPipeline


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Churn Prediction ML Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        default="config/config.yaml",
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        "--mode",
        choices=["full", "train", "evaluate"],
        default="full",
        help=(
            "Pipeline mode: "
            "'full' runs all stages, "
            "'train' runs only ingestion + training, "
            "'evaluate' loads a saved model and evaluates it."
        ),
    )
    return parser.parse_args()


def main() -> None:
    _configure_logging()
    args = _parse_args()

    pipeline = ChurnPipeline(config_path=args.config)

    if args.mode == "full":
        scores = pipeline.run()
    elif args.mode == "train":
        pipeline.run_train()
        scores = {}
    else:
        scores = pipeline.run_evaluate()

    if scores:
        print("\n=== Evaluation Scores ===")
        for metric, value in scores.items():
            print(f"  {metric:<12}: {value:.4f}")


if __name__ == "__main__":
    main()
