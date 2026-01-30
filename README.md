# BigEarthNet Distributed Deep Learning

A distributed deep learning project for large-scale remote sensing data analysis using the BigEarthNet v2.0 dataset.

## Tech Stack

![Python](https://img.shields.io/badge/Python-3.12+-3776AB?style=flat&logo=python&logoColor=white)
![uv](https://img.shields.io/badge/uv-Package%20Manager-DE5FE9?style=flat&logo=astral&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=flat&logo=jupyter&logoColor=white)
![PyArrow](https://img.shields.io/badge/PyArrow-Data-00C7B7?style=flat&logo=apache&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Analysis-150458?style=flat&logo=pandas&logoColor=white)
![AWS S3](https://img.shields.io/badge/AWS%20S3-Storage-569A31?style=flat&logo=amazons3&logoColor=white)
![Boto3](https://img.shields.io/badge/Boto3-SDK-FF9900?style=flat&logo=amazonaws&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Deep%20Learning-FF6F00?style=flat&logo=tensorflow&logoColor=white)
![Petastorm](https://img.shields.io/badge/Petastorm-Data%20Pipeline-00ADD8?style=flat&logo=apache&logoColor=white)

s3+petastorm + tensorflow pipeline 

<img width="1291" height="371" alt="image" src="https://github.com/user-attachments/assets/6e2ebb0c-e5a4-4865-913b-c46d39f2cd7d" />


## About the Project

This project explores distributed computing techniques for deep learning on big data using the BigEarthNet v2.0 dataset. BigEarthNet is a large-scale benchmark archive for remote sensing image analysis, consisting of 549,488 pairs of Sentinel-1 and Sentinel-2 satellite image patches covering various European countries.

## About BigEarthNet Dataset

**BigEarthNet v2.0** is a benchmark dataset developed by TU Berlin, consisting of:

- Coverage of 10 European countries (Austria, Belgium, Finland, Ireland, Kosovo, Lithuania, Luxembourg, Portugal, Serbia, and Switzerland)
- Images acquired between June 2017 and May 2018
- Multi-label land-cover classification based on CORINE Land Cover 2018 (CLC2018)
- Pixel-level reference maps
- **Total Files**: 8,242,325 files | 144.1 GB | TIF format

### Dataset Components

- **BigEarthNet-S2**: Sentinel-2 multispectral image patches 
- **BigEarthNet-S1**: Sentinel-1 SAR image patches 
- **Reference Maps**: Pixel-level land-cover maps
- **Metadata**: metadata in Parquet format

For more information, visit: https://bigearth.net/

## Project Structure

```
bigearthnet-dl-distributed/
├── scripts/
│   ├── bash/
│   │   ├── convert.sh           # Convert data for multiple fractions and executors
│   │   └── train.sh             # Train on multiple datasets with varying GPUs
│   ├── gen_metadata.py          # Generate metadata with S3 paths
│   ├── check.py                 # Validate S3 file availability
│   ├── to_petastorm.py          # Convert TIF to Petastorm format
│   ├── train.py                 # Train U-Net segmentation model
│   └── Readme.md                # Detailed script documentation
├── notebooks/                    # Jupyter notebooks for exploration
├── experiments/                  # Experiment outputs and results
├── pyproject.toml               # Project dependencies
└── README.md                    # This file
```

## Setup

### Prerequisites

- Python 3.12 or higher
- [uv](https://github.com/astral-sh/uv) - Fast Python package installer
- AWS credentials configured for S3 access
- Apache Spark (for data conversion)

### Install uv

**macOS/Linux:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows:**
```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**Using pip:**
```bash
pip install uv
```

### Install Project Dependencies

Once `uv` is installed, set up the project:


```bash
# Clone the repository
git clone https://github.com/krschap/bigearthnet-dl-distributed.git
cd bigearthnet-dl-distributed

# Install dependencies
uv sync
```

This makes the following commands available:
- uv run gen-metadata - Generate metadata with S3 paths
- uv run check-s3 - Validate file availability
- uv run train-model - Train segmentation model

Note: to_petastorm.py should be run using spark-submit (not uv run) as it uses Spark internally.

## Quick Start

### Step 1: One-Time Setup


Generate metadata with S3 file paths:

```bash
uv run gen-metadata \
  --meta s3://ubs-datasets/bigearthnet/metadata.parquet \
  --out s3://ubs-homes/erasmus/raj/dlproject/metadata_with_paths.parquet
```

You should run `scripts/check.py` to ensure all the files are present in s3 which are included in the metadata.parquet.


### Step 2: Convert Data to Petastorm Format

Option A: Convert multiple fractions using bash script

```bash
./scripts/bash/convert.sh experiment_1 client
```

This processes 1%, 5%, 10%, and 20% of the data with varying executor counts and creates:
```
s3://ubs-homes/erasmus/raj/dlproject/experiments/experiment_1/petastorm/
├── frac_0.01/
│   ├── exec_8/
│   ├── exec_5/
│   ├── exec_2/
│   └── exec_1/
├── frac_0.05/
├── frac_0.10/
└── frac_0.20/
```

Option B: Convert single percentage manually using spark-submit

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

### Step 3: Train Models

Option A: Train on all converted datasets with multiple GPU configurations

```bash
# Usage: ./scripts/bash/train.sh <experiment_name> <profile_name> <executor_count>
# Note: executor_count should match one used in conversion (default: 8)
./scripts/bash/train.sh experiment_1 train 8
```

Option B: Train on single dataset

```bash
uv run train-model \
  --data s3://ubs-homes/erasmus/raj/dlproject/testpercent/petastorm \
  --epochs 10 \
  --batch 16 \
  --lr 0.001
```

## Configuration

To modify script parameters, edit the configuration values directly in the bash scripts:

**scripts/bash/convert.sh:**
```bash
EXECUTOR_MEM="16g"       # Spark executor memory
DRIVER_MEM="8g"          # Spark driver memory
CORES="3"                # Cores per executor
TARGET_FILE_MB="50"      # Target output file size
N_EXECUTORS=(8 5 2 1)    # Number of executors to test
FRACTIONS=(0.01 0.05 0.10 0.20)  # Data fractions to process
```

**scripts/bash/train.sh:**
```bash
EPOCHS="10"              # Training epochs
BATCH_SIZE="16"          # Batch size per replica
LEARNING_RATE="0.001"    # Learning rate
FRACTIONS=(0.01 0.05 0.10 0.20)  # Data fractions to train on
GPUS=(4 3 2 1)           # GPU counts to test
```

## Data Organization

The pipeline creates the following S3 structure:

```
s3://ubs-homes/erasmus/raj/dlproject/
├── metadata_with_paths.parquet
└── experiments/
    └── experiment_1/
        └── petastorm/
            ├── frac_0.01/
            │   └── exec_8/
            │       ├── train/           # Training data
            │       ├── validation/      # Validation data
            │       ├── test/            # Test data
            │       └── profile/
            │           ├── conversion_profile.json
            │           ├── conversion_profile.log
            │           ├── train_profile.json
            │           └── train_profile.log
            ├── frac_0.05/
            ├── frac_0.10/
            └── frac_0.20/
```

## Profiling

Both conversion and training scripts generate detailed performance profiles:

- conversion_profile.json/log: Data processing metrics, Spark configuration, timing information
- train_profile.json/log: GPU usage, training metrics, evaluation results

Profiles are saved to: {output_dir}/profile/

## Documentation

For detailed script usage, parameters, and examples, see:
- scripts/Readme.md - Comprehensive script documentation
- scripts/bash/ - Bash script examples

## Data Access

The BigEarthNet data should be organized on S3 as follows:

```
s3://your-bucket/bigearthnet/
├── metadata.parquet
├── BigEarthNet-S1/
├── BigEarthNet-S2/
└── Reference_Maps/
```

Download the dataset from: https://bigearth.net/

## Resources

- BigEarthNet Official Website: https://bigearth.net/
- Dataset Documentation: https://bigearth.net/static/documents/Description_BigEarthNet_v2.pdf
- BigEarthNet Pipeline: https://github.com/rsim-tu-berlin/bigearthnet-pipeline
- uv Documentation: https://docs.astral.sh/uv/