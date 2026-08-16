from fastapi import FastAPI
from pydantic import BaseModel, ValidationError

api = FastAPI()

# consider separation / loading, joblib loading + data pipe

# load model, embs model
# figure out how to cache embs


class PredictionRequest(BaseModel):
    title: str
    description: str
    # datetime? here or there? here pbb better


class PredictionResponse(BaseModel):
    label: str
    probability: float


def try_to_convert_to_preds_request(
    request: dict[str, str],
) -> PredictionRequest | None:
    try:
        return PredictionRequest(**request)
    except ValidationError as e:
        print(e.errors())
        return None


# set up answer response action
# integrate inference
@api.post("/predict", response_model=PredictionResponse)
def predict(request_raw: dict[str, str]) -> PredictionResponse:
    print(f"request: {request_raw}")

    request = try_to_convert_to_preds_request(request_raw)

    if request is None:
        return PredictionResponse(label="error", probability=0.5)

    return PredictionResponse(
        label=request.title + request.description,
        probability=0.5,
    )


def main() -> None:
    print("Hello from api!")


if __name__ == "__main__":
    main()
