from __future__ import annotations

import numpy as np
from pmlb import fetch_data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from .config import DATASETS


def load_dataset(display_name: str):
    if display_name not in DATASETS:
        raise KeyError(f"Unknown dataset: {display_name}")
    X, y = fetch_data(DATASETS[display_name], return_X_y=True)
    X = np.asarray(X)
    y = np.asarray(y).astype(int)
    return X, y


def split_and_scale(X, y, seed: int):
    """60/20/20 stratified train/DSEL/test split with train-only MinMax fitting."""
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.40, stratify=y, random_state=seed
    )
    X_dsel, X_test, y_dsel, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=seed
    )

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_dsel = scaler.transform(X_dsel)
    X_test = scaler.transform(X_test)

    return X_train, X_dsel, X_test, y_train, y_dsel, y_test
