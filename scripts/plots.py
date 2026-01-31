#!/usr/bin/env python3
"""
Script to generate speedup curves for the preprocessing  and distributed training experiments.

To run the script, use:
    python plots.py --bucket <s3-bucket-name> --output-dir <output-directory>
"""

# ------------
# IMPORTS
# ------------
import argparse
import boto3
import json
import os
import pandas as pd
import re
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Callable

# ----------------
# S3 UTILITIES
# ----------------
# load json log files from S3
def load_json_from_s3(bucket: str, key: str) -> dict:
    """
    Load a JSON file from S3.
    args:
        bucket: S3 bucket name
        key: S3 object key
    returns:
        Parsed JSON content as a dictionary
    """
    s3 = boto3.client("s3")
    obj = s3.get_object(Bucket=bucket, Key=key)
    return json.loads(obj["Body"].read())

# list json files in S3
def list_jsons_in_s3(bucket, path, suffix):
    """
    Iterate over JSON files in an S3 path.
    args:
        bucket: S3 bucket name
        path: S3 path prefix
        suffix: File suffix to filter
    """
    s3 = boto3.client("s3")
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=path):
        for obj in page.get("Contents", []):
            if obj["Key"].endswith(suffix):
                yield obj["Key"]

# create dataframe from the S3 json files
def create_dataframe_from_s3(bucket, path, suffix, parser):
    """
    Create a DataFrame from JSON files in S3.
    args:
        bucket: S3 bucket name
        path: S3 path prefix
        suffix: File suffix to filter
        parser: Function to parse each JSON file
    returns:
        DataFrame with parsed records
    """
    records = []
    for key in list_jsons_in_s3(bucket, path, suffix):
        record = parser(key)
        if record:
            records.append(record)
    return pd.DataFrame(records)



# ------------------------
# Conversion/Preprocessing Record Parser
# ------------------------
def parse_conversion_record(bucket, key) :
    """
    Parse a conversion/preprocessing record.
    args:
        bucket: S3 bucket name
        key: S3 object key
    returns:
        Parsed record as a dictionary or None if invalid
    """
    data = load_json_from_s3(bucket, key)
    parts = key.split("/")

    # Extract fraction and executor count from the key
    fraction = None
    executor = None
    for p in parts:
        if p.startswith("frac_"):
            fraction = float(p.replace("frac_", ""))
        if p.startswith("exec_"):
            executor = int(p.replace("exec_", ""))

    if fraction is None or fraction < 0.001 or executor is None:
        return None

    # Extract step durations
    steps = {s["name"]: s for s in data["steps"]}

    return {
        "fraction": fraction,
        "executor": executor,
        "total_time": data["summary"]["total_duration"],
        "read_metadata": steps["read_metadata"]["duration"],
        "spark_init": steps["spark_init"]["duration"],
        "write_train": steps["write_train"]["duration"],
        "write_validation": steps["write_validation"]["duration"],
        "write_test": steps["write_test"]["duration"],
    }

# ------------------------
# Training Record Parser for Batchsize Scaling Experiment
# ------------------------
def parse_training_petastorm(bucket, key):
    """
    Parse a training record from petastorm experiment (batch size scaling).
    args:
        bucket: S3 bucket name
        key: S3 object key
    returns:
        Parsed record as a dictionary or None if invalid
    """
    if "/exec_8/" not in key:
        return None

    data = load_json_from_s3(bucket, key)

    # Extract fraction from the key
    fraction = None
    for part in key.split("/"):
        if part.startswith("frac_"):
            fraction = float(part.replace("frac_", ""))
            break

    if fraction is None or fraction < 0.001:
        return None

    #  Extract number of GPUs from the key
    gpu_match = re.search(r"train_gpu(\d+)_profile\.json", key)
    if not gpu_match:
        return None
    num_gpus = int(gpu_match.group(1))

    # Extract step durations
    steps = {s["name"]: s for s in data["steps"]}

    # return logged record
    return {
        "fraction": fraction,
        "num_gpus": num_gpus,
        "batch_size": data["summary"]["global_batch_size"],
        "total_time": data["summary"]["total_duration"],
        "train_time": steps["training"]["duration"],
        "test_time": steps["evaluation"]["duration"],
        "test_accuracy": data["summary"]["test_accuracy"],
    }

# ------------------------
# Training Record Parser for LR Scaling Experiment
# ------------------------
def parse_training_local(bucket, key):
    """
    Parse a training record from local experiment (linear rate scaling).
    args:
        bucket: S3 bucket name
        key: S3 object key
    returns:
        Parsed record as a dictionary or None if invalid
    """
    data = load_json_from_s3(bucket, key)

    # Extract fraction and number of GPUs from the key
    fraction_match = re.search(r"frac_([0-9.]+)", key)
    gpu_match = re.search(r"gpu(\d+)_profile\.json", key)

    if not fraction_match or not gpu_match:
        return None

    fraction = float(fraction_match.group(1))
    num_gpus = int(gpu_match.group(1))

    steps = {s["name"]: s for s in data.get("steps", [])}
    summary = data.get("summary", {})

    if "training" not in steps or "evaluation" not in steps:
        return None

    return {
        "fraction": fraction,
        "num_gpus": num_gpus,
        "batch_size": summary.get("global_batch_size"),
        "learning_rate": summary.get("learning_rate"),
        "total_time": summary.get("total_duration"),
        "train_time": steps["training"]["duration"],
        "test_time": steps["evaluation"]["duration"],
        "test_accuracy": summary.get("test_accuracy")
    }

# ------------------------
# SPEEDUP COMPUTATION AND PLOTTING
# ------------------------
def calculate_speedup(df,baseline_column,baseline_value,time_column = "total_time"):
    """
    Calculate speedup relative to a baseline.
    args:
        df: DataFrame with timing data
        baseline_column: Column to identify baseline configurations
        baseline_value: Value in baseline_column to use as baseline
        time_column: Column with timing data
    returns:
        DataFrame with an additional 'speedup' column
    """
    baseline_times = df[df[baseline_column] == baseline_value].set_index("fraction")[time_column]
    df = df.copy()
    df["speedup"] = df["fraction"].map(baseline_times) / df[time_column]  # Calculate speedup
    return df

def plot_speedup(df,x_column,x_label,title, output_file, x_ticks = None):
    """
    Plot speedup curves grouped by fraction.
    args:
        df: DataFrame with speedup data
        x_column: Column for x-axis values
        x_label: Label for x-axis
        title: Plot title
        output_file: File path to save the plot
        x_ticks: Optional list of x-axis ticks
    """
    plt.figure(figsize=(8, 6))

    # Plot speedup curves for each fraction
    for fraction, group in df.groupby("fraction"):
        group = group.sort_values(x_column)
        plt.plot(
            group[x_column],
            group["speedup"],
            marker="o",
            linewidth=1,
            label=f"{fraction*100:.0f}%"
        )

    # Plot ideal speedup line
    max_x = df[x_column].max()
    ideal_x = list(range(1, max_x + 1))
    plt.plot(
        ideal_x,
        ideal_x,
        linestyle="--",
        linewidth=1,
        label="Ideal Speedup"
    )

    plt.xlabel(x_label)
    if x_ticks:
        plt.xticks(x_ticks)
    plt.ylabel("Speedup")
    plt.title(title)
    plt.grid(True)
    plt.legend(title="Sample Size")
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

# -------------------------
# RUN EXPERIMENT
# -------------------------
def run_experiment(bucket, output_dir, config, parser) :
    """
    Run a single experiment and generate its speedup plot.
    args:
        bucket: S3 bucket name
        output_dir: Directory to save output files
        config: Experiment configuration dictionary
        parser: Function to parse each JSON file
    """
    # Create dataframe from S3
    df = create_dataframe_from_s3(bucket, config["path"], config["suffix"], lambda key: parser(bucket, key))
    df = df.sort_values(["fraction", config["baseline_column"]]) # Sort dataframe
    df = calculate_speedup(df, config["baseline_column"], config["baseline_value"]) # Calculate speedup
    output_file = os.path.join(output_dir, config["output_file"])
    plot_speedup(df,
                 x_column=config["baseline_column"],
                 x_label=config["x_label"],
                 title=config["title"],
                 output_file=output_file,
                 x_ticks=config.get("x_ticks"),
        )

# -------------------------
# MAIN ARGUMENTS
# -------------------------
def main():
    """
    Main execution function.
    """
    parser = argparse.ArgumentParser(description="Generate speedup curves.")
    parser.add_argument("--bucket", type=str, default="ubs-homes", help="S3 bucket name")
    parser.add_argument("--output-dir", type=str, default="results", help="Output directory for plots")
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    experiments = [
        # Preprocessing/Conversion Experiment
        {
            "path": "erasmus/raj/dlproject/experiments/experiment_min_default/",
            "suffix": "conversion_profile.json",
            "baseline_column": "executor",
            "baseline_value": 1,
            "x_label": "Number of Executors",
            "title": "Preprocessing Speedup (Conversion)",
            "output_file": "speedup_preprocessing.png",
            "x_ticks": [1, 2, 5, 8],
            "parser": parse_conversion_record,
        },
        # Batch Size Scaling Training Experiment
        {
            "path": "erasmus/raj/dlproject/experiments/experiment_final/petastorm/",
            "suffix": "_profile.json",
            "baseline_column": "num_gpus",
            "baseline_value": 1,
            "x_label": "Number of GPUs",
            "title": "Training Speedup (Batch Size Scaling)",
            "output_file": "speedup_training_batchsize.png",
            "x_ticks": [1, 2, 3, 4],
            "parser": parse_training_petastorm,
        },
        # Linear Rate Scaling Training Experiment
        {
            "path": "erasmus/raj/dlproject/experiments/experiment_local/",
            "suffix": "_profile.json",
            "baseline_column": "num_gpus",
            "baseline_value": 1,
            "x_label": "Number of GPUs",
            "title": "Training Speedup (Linear Rate Scaling)",
            "output_file": "speedup_training_lr.png",
            "x_ticks": [1, 2, 3, 4],
            "parser": parse_training_local,
        },
    ]

    for exp in experiments:
        run_experiment(args.bucket, args.output_dir, exp, exp["parser"])

if __name__ == "__main__":
    main()
