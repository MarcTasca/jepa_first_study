from src.config import ExperimentConfig
from src.runner import Runner


def main():
    config = ExperimentConfig.from_args()
    runner = Runner(config)
    runner.run()


if __name__ == "__main__":
    main()
