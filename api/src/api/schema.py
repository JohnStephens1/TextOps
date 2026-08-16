from pydantic import BaseModel


# TODO consider datetime impl
class PredictionRequest(BaseModel):
    title: str
    description: str


class PredictionResponse(BaseModel):
    label: str
    probability: float
