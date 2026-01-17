# Scripts

End-to-end pipeline for BigEarthNet satellite imagery processing and model training.

## Overview

This project provides a complete workflow for processing BigEarthNet satellite data and training semantic segmentation models:

1. gen_metadata.py - Generate metadata with S3 paths (one-time setup)
2. check.py - Validate file availability on S3 (optional verification)
3. to_petastorm.py - Convert TIF files to Petastorm format
4. train.py - Train U-Net segmentation model

## Quick Start

### One-Time Setup Tasks

These scripts only need to be run once to prepare your data:

**1. Generate Metadata with S3 Paths**

Adds S3 file paths to the BigEarthNet metadata parquet file.

```bash
uv run gen-metadata \
  --meta s3://ubs-datasets/bigearthnet/metadata.parquet \
  --out s3://ubs-homes/erasmus/raj/dlproject/metadata_with_paths.parquet
```

Parameters:
- --meta: Input metadata parquet path from BigEarthNet dataset
- --out: Output path for enhanced metadata with S3 paths

**2. Validate Files (Optional)**

Verifies that all required files exist on S3.

```bash
uv run check-s3 \
  --meta s3://ubs-homes/erasmus/raj/dlproject/metadata_with_paths.parquet \
  --out s3://ubs-homes/erasmus/raj/dlproject/validation.json \
  --frac 0.001 \
  --workers 50
```

Parameters:
- --meta: Metadata parquet with S3 file paths
- --out: Output JSON file with validation results
- --frac: Fraction of data to validate (0.001 = 0.1%)
- --workers: Number of parallel workers

### Main Workflow

**Option A: Using Bash Scripts (Recommended)**

Run complete conversion pipeline for multiple data percentages:

```bash
# Convert data to Petastorm format
./scripts/bash/run_conversion.sh experiment_1

# Train models on converted data
./scripts/bash/run_training.sh experiment_1
```

The conversion script creates this structure:
```
s3://ubs-homes/erasmus/raj/dlproject/experiments/
└── experiment_1/
    └── petastorm/
        ├── 1percent/
        │   ├── train/
        │   ├── validation/
        │   ├── test/
        │   └── profile/
        │       ├── conversion_profile.json
        │       └── conversion_profile.log
        ├── 3percent/
        ├── 5percent/
        ├── 7percent/
        └── 10percent/
```

Training adds profile data to each percentage folder:
```
petastorm/1percent/profile/
├── conversion_profile.json
├── conversion_profile.log
├── train_profile.json
└── train_profile.log
```

Configuration:

To modify script parameters, edit the configuration values directly in the bash scripts:

**run_conversion.sh:**
```bash
EXECUTOR_MEM="8g"        # Spark executor memory
DRIVER_MEM="4g"          # Spark driver memory
CORES="4"                # Cores per executor
N_EXECUTORS="3"          # Number of executors
SPARK_PACKAGES="ch.cern.sparkmeasure:spark-measure_2.12:0.27"
```

**run_training.sh:**
```bash
EPOCHS="10"              # Training epochs
BATCH_SIZE="16"          # Batch size per replica
LEARNING_RATE="0.001"    # Learning rate
```

Then run:
```bash
./scripts/bash/run_conversion.sh experiment_1
./scripts/bash/run_training.sh experiment_1
```

**Option B: Manual Single Run**

Convert specific percentage of data using spark-submit:

```bash
spark-submit \
  --master yarn \
  --deploy-mode client \
  --packages ch.cern.sparkmeasure:spark-measure_2.12:0.27 \
  scripts/to_petastorm.py \
  --meta s3://ubs-homes/erasmus/raj/dlproject/metadata_with_paths.parquet \
  --out s3://ubs-homes/erasmus/raj/dlproject/testpercent/petastorm \
  --frac 0.01 \
  --executor-mem 8g \
  --driver-mem 4g \
  --core 4 \
  --n_executor 3
```

Train model on converted data:

```bash
uv run train-model \
  --data s3://ubs-homes/erasmus/raj/dlproject/testpercent/petastorm \
  --epochs 10 \
  --batch 16 \
  --lr 0.001
```

## Script Parameters

### gen_metadata.py

```
--meta    Input metadata parquet path (default: s3://ubs-datasets/bigearthnet/metadata.parquet)
--out     Output path for augmented metadata (default: s3://ubs-homes/erasmus/raj/dlproject/metadata_with_paths.parquet)
```

### check.py

```
--meta     Metadata parquet with file paths
--out      Output path for validation results (JSON)
--frac     Data fraction to check, 0.0-1.0 (default: 0.001)
--workers  Parallel workers for S3 checks (default: 50)
```

### to_petastorm.py

```
--meta         Metadata parquet path
--out          Output directory for Petastorm dataset
--frac         Data fraction to process, 0.0-1.0 (default: 0.001)
--executor-mem Spark executor memory (default: 4g)
--driver-mem   Spark driver memory (default: 4g)
--core         Executor cores (default: 2)
--n_executor   Number of executors (default: 2)
```

### train.py

```
--data    Petastorm dataset path (contains train/validation/test subdirs)
--epochs  Number of training epochs (default: 5)
--batch   Batch size per replica (default: 16)
--lr      Learning rate (default: 0.001)
```


## Profiling

Both conversion and training scripts generate detailed profiles:

**conversion_profile.json/log**
- Metadata loading time
- Data splitting and sampling time
- Spark initialization time
- Per-split processing time
- Total conversion time
- Dataset statistics

**train_profile.json/log**
- GPU setup time
- Path verification time
- Strategy initialization time
- Dataset loading time
- Model building time
- Training time
- Evaluation time
- Performance metrics

Access profiles at: {output_dir}/profile/

## Example Workflow

Complete pipeline example for 1% of dataset:

```bash
# Step 1: Generate metadata (one-time)
uv run gen-metadata \
  --meta s3://ubs-datasets/bigearthnet/metadata.parquet \
  --out s3://ubs-homes/erasmus/raj/dlproject/metadata_with_paths.parquet

# Step 2: Convert to Petastorm using spark-submit
spark-submit \
  --master yarn \
  --deploy-mode client \
  --packages ch.cern.sparkmeasure:spark-measure_2.12:0.27 \
  scripts/to_petastorm.py \
  --meta s3://ubs-homes/erasmus/raj/dlproject/metadata_with_paths.parquet \
  --out s3://ubs-homes/erasmus/raj/dlproject/testpercent/petastorm \
  --frac 0.01 \
  --executor-mem 8g \
  --driver-mem 4g \
  --core 4 \
  --n_executor 3

# Step 3: Train model
uv run train-model \
  --data s3://ubs-homes/erasmus/raj/dlproject/testpercent/petastorm \
  --epochs 10 \
  --batch 16 \
  --lr 0.001
```

Or use the automated scripts:

```bash
# Convert multiple percentages
./scripts/bash/run_conversion.sh experiment_1

# Train on all converted data
./scripts/bash/run_training.sh experiment_1
```
