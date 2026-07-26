from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any

from scipy.stats import loguniform, randint, uniform  # type: ignore
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


class Model(ABC):
    @abstractmethod
    def get_default_params(self) -> dict[str, Any]:
        pass

    @abstractmethod
    def get_param_distribution(self) -> dict[str, Any]:
        pass

    @abstractmethod
    def get_model(self) -> BaseEstimator:
        pass


def prefix_dict_keys_with_model(dic: dict[str, Any]) -> dict[str, Any]:
    return {f"model__{k}": v for k, v in dic.items()}


class RandomForestModel(Model):
    def __init__(self) -> None:
        super().__init__()
        self.default_params = {
            "n_estimators": 10,
            "max_depth": 5,
            "min_samples_split": 3,
            "min_samples_leaf": 2,
            "max_features": "sqrt",
        }
        self.default_param_dist = {
            "n_estimators": [10],
            "max_depth": [5, 10],
            "min_samples_split": [3, 8],
            "min_samples_leaf": [2, 4],
            "max_features": ["sqrt"],
        }
        self.model: RandomForestClassifier

    def get_default_params(self, add_prefix_for_pipe: bool = True) -> dict[str, Any]:
        return (
            prefix_dict_keys_with_model(self.default_params)
            if add_prefix_for_pipe
            else self.default_params
        )

    def get_param_distribution(
        self, add_prefix_for_pipe: bool = True
    ) -> dict[str, Any]:
        return (
            prefix_dict_keys_with_model(self.default_param_dist)
            if add_prefix_for_pipe
            else self.default_param_dist
        )

    def get_model(self):
        return RandomForestClassifier()


def get_xgboost_param_distribution() -> dict[str, Any]:
    return {
        # Tree complexity
        "model__max_depth": randint(3, 11),
        "model__min_child_weight": randint(1, 10),
        # Learning
        "model__learning_rate": loguniform(1e-3, 3e-1),
        "model__n_estimators": randint(100, 1000),
        # Row/column sampling
        "model__subsample": uniform(0.5, 0.5),  # 0.5 - 1.0
        "model__colsample_bytree": uniform(0.5, 0.5),  # 0.5 - 1.0
        # Regularization
        "model__gamma": uniform(0, 5),
        "model__reg_alpha": loguniform(1e-4, 10),
        "model__reg_lambda": loguniform(1e-3, 100),
    }


def get_model_XGBClassifier(
    model_params: Mapping[str, Any] | None = None,
) -> XGBClassifier:
    if model_params is None:
        model_params = {}

    model = XGBClassifier(**model_params)

    return model


def get_param_dist_random_forest():
    return dict(  # noqa: C408
        model__n_estimators=[10],
        model__max_depth=[5, 10],
        model__min_samples_split=[3, 8],
        model__min_samples_leaf=[2, 4],
        model__max_features=["sqrt"],
    )


def get_model_random_forest():
    return RandomForestClassifier()
