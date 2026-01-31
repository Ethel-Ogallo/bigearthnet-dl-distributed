#!/bin/bash

# ---------------------------------------------
# BigEarthNet Data Conversion Script
# ---------------------------------------------
# Description:
# Automates conversion of BigEarthNet TIFs to Petastorm format.
# Supports multiple executor configurations and fractions of dataset.
# Creates structured output folders:
# s3://.../experiments/<experiment_name>/petastorm/frac_<fraction>/exec_<n_executors>
# ---------------------------------------------

# CLI Arguments and Default Values
METADATA_PATH="s3a://ubs-homes/erasmus/raj/dlproject/metadata_with_paths.parquet" # Path to input metadata with S3 paths
ROOT_DIR="s3a://ubs-homes/erasmus/raj/dlproject/experiments" # Root directory for experiments
EXPERIMENT_NAME="${1:-experiment_final}" # Name of the experiment (default: experiment_final)
DEPLOY_MODE="${2:-cluster}" # Spark deploy mode (default: cluster)
OUTPUT_BASE="${ROOT_DIR}/${EXPERIMENT_NAME}/petastorm" # Base output folder for this experiment


# Spark Configuration
EXECUTOR_MEM="16g"       # Memory per Spark executor
DRIVER_MEM="8g"          # Memory for Spark driver
CORES="3"                # Number of cores per executor
TARGET_FILE_MB="50"      # Target file size for Petastorm parquet output


# Iteration parameters
N_EXECUTORS=(8 5 2 1)    # Number of Spark executors to test
FRACTIONS=(0.01 0.05 0.10 0.20)  # Fraction of dataset to convert

# Start message
echo "Starting Conversion: ${EXPERIMENT_NAME}"
echo "Output Base: ${OUTPUT_BASE}"


# Loop over number of executors
for n_exec in "${N_EXECUTORS[@]}"; do
    echo "Number of Executors: ${n_exec}"

    # Loop over fractions of dataset
    for frac in "${FRACTIONS[@]}"; do
        # Output folder for this fraction and executor config
        FOLDER_NAME="frac_${frac}"
        OUTPUT_DIR="${OUTPUT_BASE}/${FOLDER_NAME}/exec_${n_exec}"

        # Profile name (optional) for logging in Python script
        # P_NAME="conversion_${frac}"

        echo "Processing fraction=${frac}, executors=${n_exec} -> ${OUTPUT_DIR}"

        # Submit PySpark job to convert TIFs to Petastorm
        spark-submit \
            --master yarn \
            --deploy-mode "${DEPLOY_MODE}" \
            scripts/to_petastorm.py \
            --meta "${METADATA_PATH}" \
            --out "${OUTPUT_DIR}" \
            --frac "${frac}" \
            --executor-mem "${EXECUTOR_MEM}" \
            --driver-mem "${DRIVER_MEM}" \
            --core "${CORES}" \
            --n_executor "${n_exec}" \
            --target-file-mb "${TARGET_FILE_MB}"
            # --p_name "${P_NAME}"  # Uncomment if you want per-fraction profile name

        # Check return status and handle errors
        if [ $? -eq 0 ]; then
            echo "Success: fraction=${frac}, executors=${n_exec}"
        else
            echo "ERROR: Failed fraction=${frac}, executors=${n_exec}"
            exit 1  # Stop script on failure
        fi
    done
done
