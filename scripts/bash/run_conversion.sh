#!/bin/bash

# BigEarthNet Data Conversion Script
# Converts TIF data to Petastorm format for different data percentages
# Creates folder structure: s3://ubs-homes/erasmus/raj/dlproject/experiments/experiment_1/petastorm/{percentage}

# Base configuration
METADATA_PATH="s3a://ubs-homes/erasmus/raj/dlproject/metadata_with_paths.parquet"
ROOT_DIR="s3://ubs-homes/erasmus/raj/dlproject/experiments"
EXPERIMENT_NAME="${1:-experiment_1}"
OUTPUT_BASE="${ROOT_DIR}/${EXPERIMENT_NAME}/petastorm"

# Spark configuration
EXECUTOR_MEM="8g"
DRIVER_MEM="4g"
CORES="4"
N_EXECUTORS="3"
SPARK_PACKAGES="ch.cern.sparkmeasure:spark-measure_2.12:0.27"

# Data fraction to process
FRACTIONS=(0.01 0.03 0.05 0.07 0.10)

echo "Starting BigEarthNet conversion pipeline"
echo "Experiment: ${EXPERIMENT_NAME}"
echo "Output base: ${OUTPUT_BASE}"
echo "Spark Config: Executors=${N_EXECUTORS}, Cores=${CORES}, Executor Memory=${EXECUTOR_MEM}, Driver Memory=${DRIVER_MEM}"
echo "Spark Packages: ${SPARK_PACKAGES}"
echo ""

# Run for each fraction
for frac in "${FRACTIONS[@]}"; do
    FOLDER_NAME="frac_${frac}"
    OUTPUT_DIR="${OUTPUT_BASE}/${FOLDER_NAME}"

    echo "Processing fraction=${frac}"
    echo "Output: ${OUTPUT_DIR}"

    spark-submit \
        --master yarn \
        --deploy-mode client \
        --packages "${SPARK_PACKAGES}" \
        scripts/to_petastorm.py \
        --meta "${METADATA_PATH}" \
        --out "${OUTPUT_DIR}" \
        --frac "${FRACTION}" \
        --executor-mem "${EXECUTOR_MEM}" \
        --driver-mem "${DRIVER_MEM}" \
        --core "${CORES}" \
        --n_executor "${N_EXECUTORS}" \
        > /dev/null 2>&1
    
    if [ $? -eq 0 ]; then
        echo "Successfully completed conversion for ${pct}%"
        echo ""
    else
        echo "ERROR: Conversion failed for ${pct}%"
        exit 1
    fi
done

echo "All conversions completed successfully!"
echo "Results available at: ${OUTPUT_BASE}"