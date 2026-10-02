from __future__ import annotations

import logging
import time
import tracemalloc
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from .config import STATIC_METHODS
from .data import load_dataset, split_and_scale
from .models import build_dynamic_models, build_static_models, train_base_pool

LOGGER = logging.getLogger(__name__)


def _fit_method(name, model, X_train, y_train, X_dsel, y_dsel):
    if name in STATIC_METHODS:
        model.fit(X_train, y_train)
        return

    if hasattr(model, "fit"):
        model.fit(X_dsel, y_dsel)
        return

    if hasattr(model, "partial_fit"):
        # Stream-oriented methods are fed DSEL in order. This is explicit rather
        # than hidden in the benchmark loop so the protocol is auditable.
        model.partial_fit(X_dsel, y_dsel)
        return

    raise TypeError(f"{name} exposes neither fit() nor partial_fit().")


def _predict_method(model, X_test):
    try:
        return np.asarray(model.predict(X_test))
    except (ValueError, TypeError, IndexError, AttributeError):
        # Some reimplemented methods accept one query at a time.
        return np.asarray([model.predict(np.asarray([x]))[0] for x in X_test])


def run_benchmark(
    *,
    dataset_names,
    method_names,
    seeds,
    classifier_templates,
    k,
    output_dir: Path,
    fail_fast: bool = False,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / "raw_results.csv"
    rows = []

    for seed in seeds:
        LOGGER.info("Seed %s", seed)
        np.random.seed(seed)

        for dataset_name in dataset_names:
            LOGGER.info("Dataset: %s", dataset_name)
            try:
                X, y = load_dataset(dataset_name)
                X_train, X_dsel, X_test, y_train, y_dsel, y_test = split_and_scale(X, y, seed)
            except Exception as exc:
                LOGGER.exception("Dataset setup failed: %s", dataset_name)
                if fail_fast:
                    raise
                for method in method_names:
                    rows.append({
                        "seed": seed, "dataset": dataset_name, "method": method,
                        "status": "error", "error": f"dataset_setup: {type(exc).__name__}: {exc}",
                        "f1_macro": np.nan, "train_time_s": np.nan,
                        "test_time_s": np.nan, "memory_mb": np.nan,
                        "base_train_time_s": np.nan, "base_memory_mb": np.nan,
                    })
                pd.DataFrame(rows).to_csv(raw_path, index=False)
                continue

            # Shared pool: measured separately and excluded from DES method-level fit time.
            tracemalloc.start()
            base_start = time.perf_counter()
            try:
                trained_pool = train_base_pool(classifier_templates, X_train, y_train)
            finally:
                base_train_time = time.perf_counter() - base_start
                _, base_peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()
            base_memory_mb = base_peak / 1024**2

            dynamic = build_dynamic_models(trained_pool, k=k, seed=seed)
            static = build_static_models(classifier_templates)
            all_models = {**dynamic, **static}

            for method_name in method_names:
                LOGGER.info("  Method: %s", method_name)
                model = all_models[method_name]
                status, error = "ok", ""
                f1 = train_time = test_time = memory_mb = np.nan

                try:
                    tracemalloc.start()
                    train_start = time.perf_counter()
                    _fit_method(method_name, model, X_train, y_train, X_dsel, y_dsel)
                    train_time = time.perf_counter() - train_start

                    test_start = time.perf_counter()
                    y_pred = _predict_method(model, X_test)
                    test_time = time.perf_counter() - test_start

                    _, peak = tracemalloc.get_traced_memory()
                    memory_mb = peak / 1024**2
                    tracemalloc.stop()

                    f1 = f1_score(y_test, y_pred, average="macro")
                except Exception as exc:
                    if tracemalloc.is_tracing():
                        tracemalloc.stop()
                    status = "error"
                    error = f"{type(exc).__name__}: {exc}"
                    LOGGER.exception("Method failed: %s / %s / seed=%s", method_name, dataset_name, seed)
                    if fail_fast:
                        raise

                rows.append({
                    "seed": seed,
                    "dataset": dataset_name,
                    "method": method_name,
                    "status": status,
                    "error": error,
                    "f1_macro": f1,
                    "train_time_s": train_time,
                    "test_time_s": test_time,
                    "memory_mb": memory_mb,
                    "base_train_time_s": base_train_time,
                    "base_memory_mb": base_memory_mb,
                    "n_train": len(y_train),
                    "n_dsel": len(y_dsel),
                    "n_test": len(y_test),
                })

                # Checkpoint after every method so long runs are never lost.
                pd.DataFrame(rows).to_csv(raw_path, index=False)

    return pd.DataFrame(rows)
