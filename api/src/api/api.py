from fastapi import FastAPI

from .schema import PredictionRequest, PredictionResponse

api = FastAPI()

# consider separation / loading, joblib loading + data pipe

# load model, embs model
# figure out how to cache embs


@api.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    print(f"request: {request}")

    return PredictionResponse(
        label=request.title + request.description,
        probability=0.5,
    )


def main() -> None:
    print("Hello from api!")


if __name__ == "__main__":
    main()
