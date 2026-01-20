#!/bin/bash

# BigEarthNet Data Conversion Script
# Creates folder structure: s3://.../experiments/experiment_1/petastorm/frac_0.01

METADATA_PATH="s3a://ubs-homes/erasmus/raj/dlproject/metadata_with_paths.parquet"
ROOT_DIR="s3a://ubs-homes/erasmus/raj/dlproject/experiments"
EXPERIMENT_NAME="${1:-experiment_1}"
DEPLOY_MODE="${2:-cluster}"

OUTPUT_BASE="${ROOT_DIR}/${EXPERIMENT_NAME}/petastorm"

# Spark configuration
EXECUTOR_MEM="16g"
DRIVER_MEM="8g"
CORES="4"
N_EXECUTORS="8"
TARGET_FILE_MB="20" 

FRACTIONS=(0.01 0.03 0.05 0.10)

echo "Starting Conversion: ${EXPERIMENT_NAME}"
echo "Output Base: ${OUTPUT_BASE}"

for frac in "${FRACTIONS[@]}"; do
    FOLDER_NAME="frac_${frac}"
    OUTPUT_DIR="${OUTPUT_BASE}/${FOLDER_NAME}"
    
    # Profile name specific to this fraction
    # P_NAME="conversion_${frac}"

    echo "Processing fraction=${frac} -> ${OUTPUT_DIR}"

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
        --n_executor "${N_EXECUTORS}" \
        --target-file-mb "${TARGET_FILE_MB}" \
        # --p_name "${P_NAME}"
    
    if [ $? -eq 0 ]; then
        echo "Success: ${frac}%"
    else
        echo "ERROR: Failed ${frac}%"
        exit 1
    fi
done