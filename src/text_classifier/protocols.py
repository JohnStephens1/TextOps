from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray


@runtime_checkable
class Predictor(Protocol):
    def predict(self, X: NDArray[np.float64]) -> NDArray[np.int64]: ...
    def predict_proba(self, X: NDArray[np.float64]) -> NDArray[np.float64]: ...
