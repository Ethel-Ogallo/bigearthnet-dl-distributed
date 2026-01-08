# BigEarthNet Data Pipeline

Three-stage pipeline for dataset preparation.

## Scripts

1. **gen_metadata** - Add S3 paths to metadata
2. **check** - Validate file existence
3. **to_peta_parquet** - Convert TIF to Parquet format

## Usage

### Script 1: Generate Metadata

```bash
uv run scripts/gen_metadata.py --meta s3://ubs-datasets/bigearthnet/metadata.parquet --out s3://ubs-homes/erasmus/raj/dlproject/metadata_with_paths.parquet
```

### Script 2: Check S3 Files

```bash
# Check all files
uv run scripts/check.py --meta s3://ubs-homes/erasmus/raj/dlproject/metadata_with_paths.parquet --out s3://ubs-homes/erasmus/raj/dlproject/results.json --workers 50

# Check 1% sample
uv run scripts/check.py --meta s3://ubs-homes/erasmus/raj/dlproject/metadata_with_paths.parquet --out results_1pct.json --frac 0.01
```

### Script 3: Combnine all data preprocess & create dataset for petastorm

```bash
# Convert all data
uv run scripts/to_peta_parquet.py --meta s3://ubs-homes/erasmus/raj/dlproject/metadata_with_paths.parquet --out s3://ubs-homes/erasmus/raj/dlproject/petastorm_data --workers 10

# Convert 1% sample
uv run scripts/to_peta_parquet.py --meta s3://ubs-homes/erasmus/raj/dlproject/metadata_with_paths.parquet --out s3://ubs-homes/erasmus/raj/dlproject/petastorm_data_1pct --frac 0.01
```

## Arguments

**Common:**
- `--meta`: S3 path to metadata parquet
- `--out`: S3 or local path for output
- `--frac`: Fraction of data (0.0-1.0), stratified by split (default: 1.0)

**check:**
- `--workers`: Parallel workers (default: 50)

**to_peta_parquet:**
- `--workers`: Parallel workers (default: 10)
- `--batch`: Batch size (default: 100)

## Workflow

```bash
# 1. Generate metadata
uv run scripts/gen_metadata.py --meta INPUT --out OUTPUT

# 2. Check files
uv run scripts/check.py --meta METADATA --out RESULTS

# 3. Convert to Parquet
uv run scripts/to_peta_parquet.py --meta METADATA --out DATA
```

Use `--frac 0.01` for testing, `--frac 1.0` for full dataset.

