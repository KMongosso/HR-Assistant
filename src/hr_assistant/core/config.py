import yaml
from typing import Any

def load_config(config_path: str) -> dict[str, Any]:
    with open(config_path, "r") as config_file:
        config = yaml.load(config_file, Loader=yaml.FullLoader)

    return config
