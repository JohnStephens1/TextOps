from typing import Any

from sklearn.base import BaseEstimator
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline


def get_model(
    model_cls: type[BaseEstimator], model_params: dict[str, Any]
) -> BaseEstimator:
    return model_cls(**model_params)


def get_cv_splitter() -> StratifiedKFold:
    return StratifiedKFold(n_splits=3, shuffle=True, random_state=42)


def get_search(
    pipe: Pipeline,
    search_cls: type[RandomizedSearchCV | GridSearchCV],
    param_dist: dict[str, Any],
    search_params: dict[str, Any],
) -> RandomizedSearchCV | GridSearchCV:
    # for random search, add n_iter=10
    # could make current default arguments
    # even pass down to splitter
    return search_cls(
        pipe,
        param_dist,
        cv=get_cv_splitter(),
        **search_params,
    )


# def get_random_search(
#     pipe: Pipeline,
#     param_distribution: dict[str, Any],
# ) -> RandomizedSearchCV:
#     return RandomizedSearchCV(
#         pipe,
#         param_distribution,
#         n_iter=10,
#         cv=get_cv_splitter(),
#         scoring="f1_macro",
#         random_state=42,
#         verbose=1,
#     )


# def get_grid_search(
#     pipe: Pipeline,
#     param_distribution: dict[str, Any],
# ) -> GridSearchCV:
#     return GridSearchCV(
#         pipe,
#         param_distribution,
#         cv=get_cv_splitter(),
#         scoring="f1_macro",
#         verbose=1,
#     )


# def get_search_full(
#     model_cls: type[BaseEstimator],
#     param_dist: dict[str, Any],
#     Search: type[RandomizedSearchCV | GridSearchCV],
#     search_params: dict[str, Any],
#     model_params: dict[str, Any] | None = None,
# ) -> RandomizedSearchCV | GridSearchCV:
#     if model_params is None:
#         model_params = {}

#     model = get_model(model_cls, model_params)
#     pipe = get_model_pipe(model)
#     search = get_search(
#         pipe,
#         Search,
#         param_dist,
#         search_params,
#     )

#     return search
