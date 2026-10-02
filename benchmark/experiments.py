#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import sys

from des_benchmark.config import ALL_METHODS, DATASETS, DEFAULT_K, DEFAULT_SEEDS, make_base_classifiers


def _csv_or_space(values):
    out = []
    for value in values:
        out.extend([x.strip() for x in value.split(",") if x.strip()])
    return out


def parse_args():
    p = argparse.ArgumentParser(
        description="Reproducible benchmark for Dynamic Ensemble Selection (DES).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--datasets", nargs="+", default=["all"], help="Dataset display names, PMLB ids, or 'all'.")
    p.add_argument("--methods", nargs="+", default=["all"], help="Method names or 'all'.")
    p.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    p.add_argument("--k", type=int, default=DEFAULT_K, help="Neighborhood size used by applicable DES methods.")
    p.add_argument("--pool-profile", choices=["current13", "full17"], default="current13")
    p.add_argument("--output", type=Path, default=Path("results"))
    p.add_argument("--fail-fast", action="store_true", help="Stop at the first method/dataset error.")
    p.add_argument("--list-datasets", action="store_true")
    p.add_argument("--list-methods", action="store_true")
    p.add_argument("--preflight", action="store_true", help="Check dependencies and print resolved configuration, then exit.")
    return p.parse_args()


def resolve_datasets(values):
    values = _csv_or_space(values)
    if values == ["all"] or "all" in values:
        return list(DATASETS.keys())

    by_id = {v: k for k, v in DATASETS.items()}
    resolved = []
    for value in values:
        if value in DATASETS:
            resolved.append(value)
        elif value in by_id:
            resolved.append(by_id[value])
        else:
            raise SystemExit(f"Unknown dataset '{value}'. Use --list-datasets.")
    return resolved


def resolve_methods(values):
    values = _csv_or_space(values)
    if values == ["all"] or "all" in values:
        return list(ALL_METHODS)
    unknown = [x for x in values if x not in ALL_METHODS]
    if unknown:
        raise SystemExit(f"Unknown method(s): {', '.join(unknown)}. Use --list-methods.")
    return values


def dependency_status():
    modules = {
        "numpy": "numpy",
        "pandas": "pandas",
        "scipy": "scipy",
        "scikit-learn": "sklearn",
        "pmlb": "pmlb",
        "deslib": "deslib",
        "tqdm": "tqdm",
        "torch": "torch",
    }
    return {pkg: importlib.util.find_spec(module) is not None for pkg, module in modules.items()}


def main():
    args = parse_args()

    if args.list_datasets:
        for name, pmlb_id in DATASETS.items():
            print(f"{name:24s}  {pmlb_id}")
        return 0

    if args.list_methods:
        for name in ALL_METHODS:
            print(name)
        return 0

    datasets = resolve_datasets(args.datasets)
    methods = resolve_methods(args.methods)
    classifiers = make_base_classifiers(args.pool_profile)

    config = {
        "datasets": datasets,
        "methods": methods,
        "seeds": args.seeds,
        "k": args.k,
        "pool_profile": args.pool_profile,
        "n_base_classifiers": len(classifiers),
        "split": "60/20/20 stratified train/DSEL/test",
        "scaling": "MinMaxScaler fit on train only",
        "primary_metric": "macro F1",
        "memory": "tracemalloc peak Python allocation during method fit+predict",
        "des_training_time": "DES method fit/adaptation only; shared base-pool training measured separately",
    }

    if args.preflight:
        print("Dependency check")
        deps = dependency_status()
        for name, ok in deps.items():
            print(f"- {name:14s}: {'OK' if ok else 'MISSING'}")
        print("\nResolved benchmark configuration")
        for key, value in config.items():
            print(f"- {key}: {value}")
        if not all(deps.values()):
            print("\nInstall missing dependencies with: pip install -r requirements.txt")
            return 1
        return 0

    # Heavy/optional imports are delayed until an actual benchmark run.
    import logging
    from des_benchmark.results import make_pretty, save_environment, summarize
    from des_benchmark.runner import run_benchmark

    args.output.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(args.output / "benchmark.log", mode="w", encoding="utf-8"),
        ],
    )
    save_environment(args.output / "run_config.json", config)

    raw = run_benchmark(
        dataset_names=datasets,
        method_names=methods,
        seeds=args.seeds,
        classifier_templates=classifiers,
        k=args.k,
        output_dir=args.output,
        fail_fast=args.fail_fast,
    )

    summary = summarize(raw)
    pretty = make_pretty(summary)
    summary.to_csv(args.output / "summary_numeric.csv", index=False)
    pretty.to_csv(args.output / "summary.csv", index=False)

    errors = raw[raw["status"] != "ok"]
    print(f"\nCompleted. Raw results: {args.output / 'raw_results.csv'}")
    print(f"Summary: {args.output / 'summary.csv'}")
    if not errors.empty:
        print(f"WARNING: {len(errors)} run(s) failed. Inspect {args.output / 'benchmark.log'} and raw_results.csv.")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
