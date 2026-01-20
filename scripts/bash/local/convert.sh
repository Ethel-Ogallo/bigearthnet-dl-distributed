#!/bin/bash

# Local Docker Spark Conversion Script
METADATA_PATH="s3a://ubs-homes/erasmus/raj/dlproject/metadata_with_paths.parquet"
ROOT_DIR="s3a://ubs-homes/erasmus/raj/dlproject/experiments"
EXPERIMENT_NAME="${1:-local_experiment}"
OUTPUT_BASE="${ROOT_DIR}/${EXPERIMENT_NAME}/petastorm"

# Local Docker Spark Resources
EXECUTOR_MEM="8g"
DRIVER_MEM="8g"
CORES="8"
N_EXECUTORS="1"
TARGET_FILE_MB="20" 


# FRACTIONS=(0.01 0.03 0.05 0.10)

FRACTIONS=(0.0001)


echo "Starting Local Conversion: ${EXPERIMENT_NAME}"
echo "Output Base: ${OUTPUT_BASE}"

for frac in "${FRACTIONS[@]}"; do
    FOLDER_NAME="frac_${frac}"
    OUTPUT_DIR="${OUTPUT_BASE}/${FOLDER_NAME}"

    echo "Processing fraction=${frac} -> ${OUTPUT_DIR}"

    sudo docker exec spark-master spark-submit \
        --master spark://spark-master:7077 \
        scripts/to_petastorm.py \
        --driver-mem "${DRIVER_MEM}" \
        --executor-mem "${EXECUTOR_MEM}" \
        --core "${CORES}" \
        --n_executor "${N_EXECUTORS}" \
        --target-file-mb "${TARGET_FILE_MB}" \
        --meta "${METADATA_PATH}" \
        --out "${OUTPUT_DIR}" \
        --frac "${frac}"

    if [ $? -eq 0 ]; then
        echo "Success: ${frac}"
    else
        echo "ERROR: Failed ${frac}"
        exit 1
    fi
done