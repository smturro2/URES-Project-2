import numpy as np
import pandas as pd


def check_input_X(X):
    if isinstance(X, pd.DataFrame):
        X = X.to_numpy()

    if not isinstance(X, np.ndarray):
        X = np.array(X)

    if len(X.shape) != 2:
        raise ValueError(f"X has shape {X.shape}. X must have a shape of 2")
    return X