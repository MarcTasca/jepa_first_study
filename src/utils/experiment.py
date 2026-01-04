import logging
import os
import time

from src.utils.config_io import save_config


def setup_experiment(name, config, base_dir="results"):
    """
    Sets up the experiment directory structure: results/<name>/<timestamp>
    Returns the experiment directory path.
    """
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    experiment_dir = os.path.join(base_dir, name, timestamp)
    os.makedirs(experiment_dir, exist_ok=True)

    # Save the config for reproducibility
    save_config(config, os.path.join(experiment_dir, "config.yaml"))

    # Setup logging
    log_file = os.path.join(experiment_dir, "training.log")

    # Configure root logger
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )

    logging.info(f"Experiment setup complete: {experiment_dir}")
    return experiment_dir
