#!/bin/bash

# ---------------------------------------------
# BigEarthNet Training Script
# ---------------------------------------------
# Description:
# Automates training of the U-Net semantic segmentation model on BigEarthNet Petastorm datasets.
# Loops over multiple fractions of data and GPU configurations.
# Uses uv run to invoke the Python training script with appropriate arguments.
# ---------------------------------------------

# CLI Arguments and Default Values
ROOT_DIR="s3://ubs-homes/erasmus/raj/dlproject/experiments" # Root directory containing experiments
EXPERIMENT_NAME="${1:-experiment_final}" # Name of the experiment (default: experiment_final)
TRAIN_PROFILE_NAME="${2:-train}" # Profile name for logging/training outputs (default: train)
EXECUTOR_COUNT="${3:-8}"        # Number of executors used during data conversion (default: 8)
DATA_BASE="${ROOT_DIR}/${EXPERIMENT_NAME}/petastorm" # Base directory for Petastorm datasets for this experiment


# Training hyperparameters
EPOCHS="10"
BATCH_SIZE="16"
LEARNING_RATE="0.001"

# Iteration parameters
FRACTIONS=(0.01 0.05 0.10 0.20)
GPUS=(4 3 2 1)

# Start message
echo "Starting Training: ${EXPERIMENT_NAME}"
echo "Looking for data in: ${DATA_BASE}"


# Loop over fractions of dataset
for pct in "${FRACTIONS[@]}"; do

    # Loop over GPU configurations
    for GPU in "${GPUS[@]}"; do

        # Construct path to dataset for current fraction and executor count
        DATA_DIR="${DATA_BASE}/frac_${pct}/exec_${EXECUTOR_COUNT}"

        echo "Training on ${DATA_DIR} with ${GPU} GPU(s)"

        # Launch training script using uv run
        uv run train-model \
            --data "${DATA_DIR}" \
            --epochs "${EPOCHS}" \
            --p_name "${TRAIN_PROFILE_NAME}_gpu${GPU}" \
            --batch "${BATCH_SIZE}" \
            --lr "${LEARNING_RATE}" \
            --gpus "${GPU}"

        # Check return status and handle errors
        if [ $? -eq 0 ]; then
            echo "Success: ${pct}% with ${GPU} GPU(s)"
        else
            echo "ERROR: Failed ${pct}% with ${GPU} GPU(s)"
            exit 1  # Stop script on failure
        fi
    done
done
