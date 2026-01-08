# Scripts

End-to-end pipeline for BigEarthNet satellite imagery processing and training.

## Overview

1. **gen_metadata.py** - Add S3 paths to metadata
2. **check.py** - Validate file availability on S3
3. **to_tfrecord.py** - Convert TIF files to TFRecord format
4. **train.py** - Train U-Net segmentation model

## Usage

### Example: 0.1% Data Pipeline

Complete workflow for processing 0.1% of the dataset:

**Step 1: Generate metadata with S3 paths**

```bash
uv run gen-metadata --meta s3://ubs-datasets/bigearthnet/metadata.parquet --out s3://ubs-homes/erasmus/raj/dlproject/metadata_with_paths.parquet
```

**Step 2: Validate files (optional)**

```bash
uv run check-s3 --meta s3://ubs-homes/erasmus/raj/dlproject/metadata_with_paths.parquet --out s3://ubs-homes/erasmus/raj/dlproject/1percent/validation.json --frac 0.001 --workers 50
```

**Step 3: Convert to TFRecord**

```bash
uv run to-tfrecord --meta s3://ubs-homes/erasmus/raj/dlproject/metadata_with_paths.parquet --out s3://ubs-homes/erasmus/raj/dlproject/1percent/tfrecords --frac 0.001 --workers 10 --batch 100
```

**Step 4: Train model**

```bash
uv run train-model --data s3://ubs-homes/erasmus/raj/dlproject/1percent/tfrecords --epochs 10 --batch 32 --lr 0.001 --save s3://ubs-homes/erasmus/raj/dlproject/1percent/model.keras
```

## Data Organization

S3 structure:

```
s3://bucket/dlproject/
├── metadata_with_paths.parquet
├── 1percent/
│   ├── validation.json
│   ├── tfrecords/
│   │   └── part-*.tfrecord
│   └── model.keras
├── 10percent/
│   ├── validation.json
│   ├── tfrecords/
│   │   └── part-*.tfrecord
│   └── model.keras
└── full/
    ├── validation.json
    ├── tfrecords/
    │   └── part-*.tfrecord
    └── model.keras
```

## Parameters

### gen_metadata.py

- `--meta` - Input metadata parquet path
- `--out` - Output path for augmented metadata

### check.py

- `--meta` - Metadata parquet with file paths
- `--out` - Output path for validation results
- `--frac` - Data fraction (0-1, default: 1.0)
- `--workers` - Parallel workers (default: 50)

### to_tfrecord.py

- `--meta` - Metadata parquet path
- `--out` - Output directory
- `--frac` - Data fraction (0-1, default: 1.0)
- `--workers` - Parallel workers (default: 10)
- `--batch` - Patches per file (default: 100)

### train.py

- `--data` - TFRecord directory (local or S3 path)
- `--epochs` - Training epochs (default: 10)
- `--batch` - Batch size (default: 32)
- `--lr` - Learning rate (default: 0.001)
- `--save` - Model output path (local or S3, .keras format)

## Data Format

**TFRecord fields:**

- `patch_id` (string)
- `s1_data` (120x120x2 float32)
- `s2_data` (120x120x12 float32)
- `label` (120x120 uint8)

**Model:** U-Net encoder-decoder with 120x120x14 input and 256 output classes.
