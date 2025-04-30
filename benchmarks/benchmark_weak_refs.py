#!/usr/bin/env python3
"""Benchmark script for testing weak reference performance in Mesa.

This script runs benchmarks to compare the performance of models using
weak references versus standard references.
"""

import argparse
import gc
import os
import pickle
import sys
import time
from datetime import UTC, datetime

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil

# Fix imports to use relative paths
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from benchmarks.configurations import configurations
from benchmarks.weak_ref_benchmark_model import WeakRefBenchmarkModel


def get_memory_usage():
    """Get the current memory usage of this process in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)  # Convert to MB


def run_benchmark(
    model_class,
    config_name,
    seeds,
    replications,
    steps,
    parameters,
    collect_memory=True,
    verbose=True,
):
    """Run a single benchmark configuration and return timing results.

    Args:
        model_class: The model class to benchmark
        config_name: Name of this configuration
        seeds: Number of different random seeds to use
        replications: Number of replications per seed
        steps: Number of model steps to run
        parameters: Dictionary of model parameters
        collect_memory: Whether to collect memory usage metrics
        verbose: Whether to print progress information

    Returns:
        Dictionary with benchmark results
    """
    if verbose:
        print(f"Running {model_class.__name__} - {config_name}...")

    # Results storage
    init_times = []
    run_times = []
    memory_usages = []

    for seed in range(seeds):
        for rep in range(replications):
            if verbose and (rep == 0 or replications > 2):
                print(f"  Seed {seed + 1}/{seeds}, Rep {rep + 1}/{replications}")

            # Force garbage collection before each run
            gc.collect()

            # Measure initialization time
            start_time = time.time()
            model = model_class(seed=seed, **parameters)
            init_time = time.time() - start_time
            init_times.append(init_time)

            # Collect initial memory usage if requested
            if collect_memory:
                initial_memory = get_memory_usage()

            # Measure run time
            start_time = time.time()
            for _ in range(steps):
                model.step()
            run_time = time.time() - start_time
            run_times.append(run_time)

            # Collect final memory usage if requested
            if collect_memory:
                final_memory = get_memory_usage()
                memory_usage = final_memory - initial_memory
                memory_usages.append(memory_usage)

    # Calculate summary statistics
    results = {
        "model": model_class.__name__,
        "config": config_name,
        "init_time_mean": np.mean(init_times),
        "init_time_std": np.std(init_times),
        "run_time_mean": np.mean(run_times),
        "run_time_std": np.std(run_times),
    }

    if collect_memory:
        results["memory_usage_mean"] = np.mean(memory_usages)
        results["memory_usage_std"] = np.std(memory_usages)

    return results


def run_weak_ref_benchmarks(save_results=True, plot_results=True, verbose=True):
    """Run benchmarks specifically for weak reference performance.

    Args:
        save_results: Whether to save results to a file
        plot_results: Whether to generate and display plots
        verbose: Whether to print progress information

    Returns:
        DataFrame with benchmark results
    """
    all_results = []
    model_class = WeakRefBenchmarkModel

    # Only benchmark configurations for the weak ref model
    model_configs = configurations.get(model_class, {})

    if not model_configs:
        print("No configurations found for WeakRefBenchmarkModel!")
        return None

    # Run each configuration
    for config_name, config in model_configs.items():
        results = run_benchmark(
            model_class=model_class,
            config_name=config_name,
            seeds=config["seeds"],
            replications=config["replications"],
            steps=config["steps"],
            parameters=config["parameters"],
            verbose=verbose,
        )
        all_results.append(results)

    # Convert to DataFrame
    df = pd.DataFrame(all_results)

    # Print summary
    if verbose:
        print("\nResults summary:")
        print(df)

    # Save results
    if save_results:
        timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        filename = f"weakref_benchmark_{timestamp}.pickle"
        with open(filename, "wb") as f:
            pickle.dump(df, f)
        if verbose:
            print(f"Results saved to {filename}")

    # Generate plots if requested
    if plot_results:
        plot_weak_ref_comparison(df)

    return df


def plot_weak_ref_comparison(df):
    """Generate plots comparing weak reference performance.

    Args:
        df: DataFrame with benchmark results
    """
    # Extract data for standard vs weak ref configurations
    standard_configs = df[df["config"].str.contains("standard")]
    weakref_configs = df[~df["config"].str.contains("standard")]

    # Create a figure with multiple subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot initialization time comparison
    labels = [
        row["config"]
        for _, row in pd.concat([standard_configs, weakref_configs]).iterrows()
    ]
    init_times = [
        row["init_time_mean"]
        for _, row in pd.concat([standard_configs, weakref_configs]).iterrows()
    ]
    init_errors = [
        row["init_time_std"]
        for _, row in pd.concat([standard_configs, weakref_configs]).iterrows()
    ]

    axes[0].bar(labels, init_times, yerr=init_errors, capsize=5)
    axes[0].set_title("Initialization Time")
    axes[0].set_ylabel("Time (seconds)")
    axes[0].set_xticklabels(labels, rotation=45, ha="right")

    # Plot run time comparison
    run_times = [
        row["run_time_mean"]
        for _, row in pd.concat([standard_configs, weakref_configs]).iterrows()
    ]
    run_errors = [
        row["run_time_std"]
        for _, row in pd.concat([standard_configs, weakref_configs]).iterrows()
    ]

    axes[1].bar(labels, run_times, yerr=run_errors, capsize=5)
    axes[1].set_title("Run Time")
    axes[1].set_ylabel("Time (seconds)")
    axes[1].set_xticklabels(labels, rotation=45, ha="right")

    plt.tight_layout()
    plt.savefig("weakref_benchmark_comparison.png")
    plt.show()


def compare_weakref_results(file1, file2):
    """Compare results from two weak reference benchmark runs.

    Args:
        file1: Filename of first benchmark results
        file2: Filename of second benchmark results
    """
    # Load data with safety checks
    try:
        with open(file1, "rb") as f:
            df1 = joblib.load(f)
    except Exception as e:
        print(f"Error loading {file1}: {e}")
        return None

    try:
        with open(file2, "rb") as f:
            df2 = joblib.load(f)
    except Exception as e:
        print(f"Error loading {file2}: {e}")
        return None

    # Merge dataframes
    df = pd.merge(df1, df2, on=["model", "config"], suffixes=("_1", "_2"))

    # Calculate percent changes
    df["init_time_pct_change"] = (
        (df["init_time_mean_2"] - df["init_time_mean_1"]) / df["init_time_mean_1"] * 100
    )
    df["run_time_pct_change"] = (
        (df["run_time_mean_2"] - df["run_time_mean_1"]) / df["run_time_mean_1"] * 100
    )

    # Function to determine improvement emojis
    def get_emoji(pct_change):
        if pct_change < -3:  # More than 3% faster
            return "🟢"  # Green circle for improvement
        elif pct_change > 3:  # More than 3% slower
            return "🔴"  # Red circle for regression
        else:
            return "🔵"  # Blue circle for insignificant change

    # Add emoji indicators
    df["init_time_emoji"] = df["init_time_pct_change"].apply(get_emoji)
    df["run_time_emoji"] = df["run_time_pct_change"].apply(get_emoji)

    # Format table for display
    table = df[
        [
            "model",
            "config",
            "init_time_emoji",
            "init_time_pct_change",
            "run_time_emoji",
            "run_time_pct_change",
        ]
    ]

    # Round percentage changes
    table["init_time_pct_change"] = table["init_time_pct_change"].round(2)
    table["run_time_pct_change"] = table["run_time_pct_change"].round(2)

    # Print table
    print(f"Comparison between {file1} and {file2}:")
    print("Positive values mean the second run was slower (worse)")
    print("Negative values mean the second run was faster (better)")
    print()
    print(table.to_string(index=False))

    # Return the dataframe for further analysis
    return df


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run weak reference benchmarks")
    parser.add_argument(
        "--no-save",
        action="store_false",
        dest="save_results",
        help="Do not save benchmark results",
    )
    parser.add_argument(
        "--no-plot",
        action="store_false",
        dest="plot_results",
        help="Do not generate plots",
    )
    parser.add_argument(
        "--quiet",
        action="store_false",
        dest="verbose",
        help="Run quietly without progress output",
    )
    parser.add_argument(
        "--compare",
        nargs=2,
        metavar=("FILE1", "FILE2"),
        help="Compare two benchmark result files",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.compare:
        compare_weakref_results(args.compare[0], args.compare[1])
    else:
        run_weak_ref_benchmarks(
            save_results=args.save_results,
            plot_results=args.plot_results,
            verbose=args.verbose,
        )
