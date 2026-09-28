from __future__ import annotations

from pathlib import Path
import json
import platform
from importlib import metadata

import numpy as np
import pandas as pd


def summarize(raw_df: pd.DataFrame) -> pd.DataFrame:
    ok = raw_df[raw_df["status"] == "ok"].copy()
    if ok.empty:
        return pd.DataFrame()

    grouped = ok.groupby(["method", "dataset"], as_index=False).agg(
        f1_mean=("f1_weighted", "mean"),
        f1_std=("f1_weighted", lambda x: np.std(x, ddof=1) if len(x) > 1 else 0.0),
        train_time_mean=("train_time_s", "mean"),
        train_time_std=("train_time_s", lambda x: np.std(x, ddof=1) if len(x) > 1 else 0.0),
        test_time_mean=("test_time_s", "mean"),
        test_time_std=("test_time_s", lambda x: np.std(x, ddof=1) if len(x) > 1 else 0.0),
        memory_mean=("memory_mb", "mean"),
        memory_std=("memory_mb", lambda x: np.std(x, ddof=1) if len(x) > 1 else 0.0),
        n_runs=("seed", "count"),
    )
    grouped["f1_mean"] *= 100.0
    grouped["f1_std"] *= 100.0
    return grouped


def make_pretty(summary_df: pd.DataFrame) -> pd.DataFrame:
    if summary_df.empty:
        return summary_df.copy()
    out = summary_df[["method", "dataset", "n_runs"]].copy()
    out["F1 (%)"] = summary_df.apply(lambda r: f"{r.f1_mean:.3f}±{r.f1_std:.3f}", axis=1)
    out["Train Time (s)"] = summary_df.apply(lambda r: f"{r.train_time_mean:.3f}±{r.train_time_std:.4f}", axis=1)
    out["Test Time (s)"] = summary_df.apply(lambda r: f"{r.test_time_mean:.3f}±{r.test_time_std:.3f}", axis=1)
    out["Memory (MB)"] = summary_df.apply(lambda r: f"{r.memory_mean:.3f}±{r.memory_std:.3f}", axis=1)
    return out


def save_environment(path: Path, config: dict):
    packages = {}
    for pkg in ["numpy", "pandas", "scikit-learn", "deslib", "pmlb", "torch", "scipy", "tqdm"]:
        try:
            packages[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            packages[pkg] = None
    payload = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": packages,
        "config": config,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
