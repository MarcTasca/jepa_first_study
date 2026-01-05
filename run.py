import argparse

from src.config import ExperimentConfig
from src.runner import Runner
from src.utils.experiment import setup_experiment


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", type=str, help="Path to YAML config")
    parser.add_argument("--experiment", type=str, default="default", help="Experiment name")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    args, remaining = parser.parse_known_args()

    if args.config:
        config = ExperimentConfig.from_yaml(args.config)
    else:
        # Fallback to legacy arg parsing
        config = ExperimentConfig.from_args()

    # Setup experiment tracking
    exp_dir = setup_experiment(args.experiment, config)

    # Inject exp_dir into config (hacky but effective for now) or pass to Runner
    # The Runner likely expects to control logging or directories.
    # We should update Runner to accept an output directory.

    runner = Runner(config, output_dir=exp_dir)
    runner.run(resume_path=args.resume)


if __name__ == "__main__":
    main()
