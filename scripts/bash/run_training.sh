#!/bin/bash

# BigEarthNet Training Script
# Trains models on Petastorm datasets created by run_conversion.sh
# Trains on different data percentages with specified hyperparameters

# Base configuration
ROOT_DIR="s3://ubs-homes/erasmus/raj/dlproject/experiments"

EXPERIMENT_NAME="${1:-experiment_1}"

DATASET_FOLDER="${2:-petastorm}"


DATA_BASE="${ROOT_DIR}/${DATASET_FOLDER}"
# Training hyperparameters
EPOCHS="10"
BATCH_SIZE="16"
LEARNING_RATE="0.001"

FRACTIONS=(0.01 0.03 0.05)


echo "Starting BigEarthNet training pipeline"
echo "Experiment: ${EXPERIMENT_NAME}"
echo "Data base: ${DATA_BASE}"
echo "Training Config: Epochs=${EPOCHS}, Batch Size=${BATCH_SIZE}, Learning Rate=${LEARNING_RATE}"
echo ""

# Train on each percentage
for pct in "${FRACTIONS[@]}"; do
    DATA_DIR="${DATA_BASE}/frac_${pct}"
    
    echo "========================================"
    echo "Training on ${pct}% dataset"
    echo "Data: ${DATA_DIR}"
    echo "========================================"
    
    uv run train-model \
        --data "${DATA_DIR}" \
        --epochs "${EPOCHS}" \
        --p_name "${EXPERIMENT_NAME}" \
        --batch "${BATCH_SIZE}" \
        --lr "${LEARNING_RATE}"
    
    if [ $? -eq 0 ]; then
        echo "Successfully completed training for ${pct}%"
        echo ""
    else
        echo "ERROR: Training failed for ${pct}%"
        exit 1
    fi
done

echo "All training runs completed successfully!"
echo "Profiles and logs available at: ${DATA_BASE}/{percentage}/profile"