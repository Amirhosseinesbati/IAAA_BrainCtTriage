import argparse
import json

from src.pipelines.pipelines import (
    nnunet_pipeline, yolo_pipeline, mls_pipeline, ich_pipeline,
    mls_strategy_pipeline,
)
from src.strategies import list_strategies, list_mls_strategies


def _print_available_strategies():
    """Print all registered ICH strategies and their descriptions."""
    strategies = list_strategies()
    print("\n📋 Available ICH strategies:")
    for s in strategies:
        print(f"   • {s['name']:12s} — {s['display_name']}")
        print(f"     {s['description'][:100]}...")
    print()


def _print_available_mls_strategies():
    """Print all registered MLS strategies and their descriptions."""
    strategies = list_mls_strategies()
    print("\n📋 Available MLS strategies:")
    for s in strategies:
        print(f"   • {s['name']:12s} — {s['display_name']}")
        print(f"     {s['description'][:100]}...")
    print()


def _validate_config_json(config_json: str) -> dict:
    """Parse and return the JSON config, exiting with an error if invalid."""
    try:
        config_parsed = json.loads(config_json)
        print(f"   Config: {json.dumps(config_parsed, indent=2)}")
        return config_parsed
    except json.JSONDecodeError as e:
        print(f"❌ Invalid JSON config: {e}")
        exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Medical AI training pipelines.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.pipelines.run_pipeline --run nnunet
  python -m src.pipelines.run_pipeline --run ich --strategy smp --config '{"architecture":"Unet","epochs":50}'
  python -m src.pipelines.run_pipeline --run ich --list-strategies
  python -m src.pipelines.run_pipeline --run mls-strategy --strategy mls_heatmap --config '{"backbone":"hrnet_w18","epochs":100}'
  python -m src.pipelines.run_pipeline --run mls-strategy --list-mls-strategies
  python -m src.pipelines.run_pipeline --run all
        """,
    )
    parser.add_argument(
        "--run", type=str, required=True,
        choices=["nnunet", "yolo", "mls", "mls-strategy", "ich", "all"],
        help="Which pipeline to execute?",
    )
    parser.add_argument(
        "--strategy", type=str, default="nnunet",
        help="ICH strategy name (for --run ich) or MLS strategy name "
             "(for --run mls-strategy). Use --list-strategies / "
             "--list-mls-strategies to see options.",
    )
    parser.add_argument(
        "--config", type=str, default="{}",
        help="JSON config for the selected strategy (for --run ich / mls-strategy).",
    )
    parser.add_argument(
        "--list-strategies", action="store_true",
        help="Print available ICH strategies and exit.",
    )
    parser.add_argument(
        "--list-mls-strategies", action="store_true",
        help="Print available MLS strategies and exit.",
    )

    args = parser.parse_args()

    if args.list_strategies:
        _print_available_strategies()
        exit(0)

    if args.list_mls_strategies:
        _print_available_mls_strategies()
        exit(0)

    if args.run == "nnunet":
        print("🚀 Launching ONLY nnU-Net Pipeline...")
        nnunet_pipeline()
    elif args.run == "yolo":
        print("🚀 Launching ONLY YOLO Pipeline...")
        yolo_pipeline()
    elif args.run == "mls":
        print("🚀 Launching ONLY Legacy MLS Pipeline...")
        mls_pipeline()
    elif args.run == "mls-strategy":
        print(f"🚀 Launching MLS Strategy Pipeline | strategy={args.strategy}")
        _validate_config_json(args.config)
        mls_strategy_pipeline(strategy_name=args.strategy, config_json=args.config)
    elif args.run == "ich":
        print(f"🚀 Launching ICH Pipeline | strategy={args.strategy}")
        # Validate JSON config early
        _validate_config_json(args.config)
        ich_pipeline(strategy_name=args.strategy, config_json=args.config)
    elif args.run == "all":
        print("🚀 Launching ALL Pipelines sequentially...")
        nnunet_pipeline()
        yolo_pipeline()
        mls_pipeline()
        mls_strategy_pipeline()