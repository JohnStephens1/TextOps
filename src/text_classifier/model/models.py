from abc import ABC
from collections.abc import Mapping
from typing import Any

from scipy.stats import loguniform, randint, uniform  # type: ignore
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


def prefix_dict_keys_with_model(dic: dict[str, Any]) -> dict[str, Any]:
    return {f"model__{k}": v for k, v in dic.items()}


class ModelBase(ABC):
    def __init__(
        self,
        default_params: dict[str, Any],
        default_param_dist: dict[str, Any],
        ModelType: Any,
        instantiate_w_default_params: bool = False,
    ) -> None:
        super().__init__()
        self.default_params = default_params
        self.default_param_dist = default_param_dist
        self.default_params_w_model_prefix = prefix_dict_keys_with_model(default_params)
        self.default_param_dist_w_model_prefix = prefix_dict_keys_with_model(
            default_param_dist
        )
        self.model = (
            ModelType(**self.default_params)
            if instantiate_w_default_params
            else ModelType()
        )


class RandomForestModel(ModelBase):
    def __init__(self, instantiate_w_default_params: bool = False) -> None:
        super().__init__(
            {
                "n_estimators": 10,
                "max_depth": 5,
                "min_samples_split": 3,
                "min_samples_leaf": 2,
                "max_features": "sqrt",
            },
            {
                "n_estimators": [20],
                "max_depth": randint(5, 10),
                "min_samples_split": randint(3, 8),
                "min_samples_leaf": randint(2, 5),
                "max_features": ["sqrt"],
            },
            RandomForestClassifier,
            instantiate_w_default_params,
        )


# def prefix_mpt_keys_with_model(
#     dic: MappingProxyType[str, Any],
# ) -> MappingProxyType[str, Any]:
#     return MappingProxyType({f"model__{k}": v for k, v in dic.items()})


# def get_model_params_dist(
#     params: MappingProxyType[str, Any], param_dist: MappingProxyType[str, Any]
# ) -> tuple[
#     MappingProxyType[str, Any],
#     MappingProxyType[str, Any],
#     MappingProxyType[str, Any],
#     MappingProxyType[str, Any],
# ]:
#     params_w_model_prefix = prefix_mpt_keys_with_model(params)
#     param_dist_w_model_prefix = prefix_mpt_keys_with_model(param_dist)

#     return params, params_w_model_prefix, param_dist, param_dist_w_model_prefix


# class ModelBase(ABC):
#     _model: Any
#     params_w_model_prefix: MappingProxyType[str, Any]

#     @classmethod
#     def get_model(cls, with_params: bool):
#         if with_params:
#             return cls._model(**cls.params_w_model_prefix)
#         return cls._model()


# class RandomForestModel(ModelBase):
#     _model = RandomForestClassifier
#     params, params_w_model_prefix, param_dist, param_dist_w_model_prefix = (
#         get_model_params_dist(
#             MappingProxyType(
#                 {
#                     "n_estimators": 10,
#                     "max_depth": 5,
#                     "min_samples_split": 3,
#                     "min_samples_leaf": 2,
#                     "max_features": "sqrt",
#                 }
#             ),
#             MappingProxyType(
#                 {
#                     "n_estimators": [10],
#                     "max_depth": [5, 10],
#                     "min_samples_split": [3, 8],
#                     "min_samples_leaf": [2, 4],
#                     "max_features": ["sqrt"],
#                 }
#             ),
#         )
#     )


# RandomForestModel.get_model(False)


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


def get_param_dist_random_forest() -> dict[str, Any]:
    return dict(  # noqa: C408
        model__n_estimators=[10],
        model__max_depth=[5, 10],
        model__min_samples_split=[3, 8],
        model__min_samples_leaf=[2, 4],
        model__max_features=["sqrt"],
    )


def get_model_random_forest() -> RandomForestClassifier:
    return RandomForestClassifier()
