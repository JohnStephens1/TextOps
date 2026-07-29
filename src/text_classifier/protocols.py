from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd
from numpy.typing import NDArray


@runtime_checkable
class Predictor(Protocol):
    def predict(self, X: pd.DataFrame) -> NDArray[np.int64]: ...
    def predict_proba(self, X: pd.DataFrame) -> NDArray[np.float64]: ...
