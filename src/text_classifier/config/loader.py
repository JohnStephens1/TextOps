from pathlib import Path
from typing import Any

import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

from text_classifier.config.config import PROJECT_ROOT


def load_raw_config(config_path: Path = PROJECT_ROOT / "params.yaml"):

    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_and_parse_config() -> dict[str, Any]:
    le_config = load_raw_config()
    target_model_dict = le_config["models"][le_config["active_model"]]

    if le_config["active_model"] == "random_forest":
        result_dict = parse_random_forest_config(target_model_dict)
    else:
        raise KeyError("active_model not defined")

    return result_dict


def parse_random_forest_config(model_dict: dict[str, Any]) -> dict[str, Any]:
    result_dict = {}

    result_dict["model_default_params"] = model_dict["default_params"]
    result_dict["model_cls"] = (
        RandomForestClassifier
        if model_dict["name_cls"] == "RandomForestClassifier"
        else KeyError(f"unexpected model class: {model_dict['name_cls']}")
    )
    result_dict["instantiate_w_default_params"] = model_dict[
        "instantiate_w_default_params"
    ]

    result_dict["search_param_dist"] = model_dict["search"]["param_dist"]
    result_dict["search_params"] = model_dict["search"]["params"]
    result_dict["search_cls"] = (
        GridSearchCV
        if model_dict["search"]["name_cls"] == "GridSearchCV"
        else KeyError(f"unexpected search class: {model_dict['search']['name_cls']}")
    )

    return result_dict
