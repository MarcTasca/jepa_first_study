from dataclasses import asdict

import yaml


def save_config(config_obj, path):
    """Saves a dataclass config object to a YAML file."""
    with open(path, "w") as f:
        yaml.dump(asdict(config_obj), f, default_flow_style=False)


def load_config(config_cls, path):
    """Loads a YAML file into a dataclass config object."""
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    return config_cls(**data)
