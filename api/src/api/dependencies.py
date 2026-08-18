from typing import Annotated

from fastapi import Depends, Request

from text_classifier.schema import PredictionResources


def get_prediction_resources(request: Request) -> PredictionResources:
    return request.app.state.prediction_resources


PredictionResourcesDeps = Annotated[
    PredictionResources, Depends(get_prediction_resources)
]
