#!/bin/bash

# BigEarthNet Training Script

ROOT_DIR="s3://ubs-homes/erasmus/raj/dlproject/experiments"
EXPERIMENT_NAME="${1:-local_experiment}"
TRAIN_PROFILE_NAME="${2:-train_local}"

DATA_BASE="${ROOT_DIR}/${EXPERIMENT_NAME}/petastorm"

EPOCHS="10"
BATCH_SIZE="16"
LEARNING_RATE="0.001"

FRACTIONS=(0.01 0.03 0.05 0.10)

echo "Starting Training: ${EXPERIMENT_NAME}"
echo "Looking for data in: ${DATA_BASE}"

for pct in "${FRACTIONS[@]}"; do
    DATA_DIR="${DATA_BASE}/frac_${pct}"

    echo "Training on ${DATA_DIR}"
    
    uv run train-model \
        --data "${DATA_DIR}" \
        --epochs "${EPOCHS}" \
        --p_name "${TRAIN_PROFILE_NAME}" \
        --batch "${BATCH_SIZE}" \
        --lr "${LEARNING_RATE}"
        --enable-cache
    
    if [ $? -eq 0 ]; then
        echo "Success: ${pct}%"
    else
        echo "ERROR: Failed ${pct}%"
        exit 1
    fi
done