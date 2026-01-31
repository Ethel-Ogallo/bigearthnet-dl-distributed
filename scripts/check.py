"""
Check BigEarthNet S3 file existence with parallel requests.
Authors: Kshitij Raj & Ethel Ogallo
Last Updated: 30-01-2026

Description:
-------------
This script validates the existence of BigEarthNet patch data stored on Amazon S3.
For each patch, it checks whether Sentinel-1, Sentinel-2, and reference data folders
exist before downstream processing.

"""

# ------------------ 
# IMPORTS
# ------------------
import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import pyarrow.parquet as pq
import s3fs
from tqdm import tqdm


# ------------------ 
# CHECK S3 FOLDER EXISTS
# ------------------
def check_s3_folder_exists(fs, s3_path):
    """Check if S3 folder exists by listing objects with the prefix.

    Why: Amazon S3 doesn't have real directories. We check if any objects exist with the path prefix.
    This validates that data was uploaded correctly before running expensive conversions.

    Parameters:
        fs : (s3fs.S3FileSystem) Initialized S3 filesystem object.
        s3_path : (str) Full S3 path to check (e.g., s3://bucket/path)

    Returns: (bool) True if objects exist under the prefix, False otherwise.
    """
    if not s3_path or not isinstance(s3_path, str) or not s3_path.startswith("s3://"):
        return False

    # Convert s3://bucket/path format to bucket/path expected by s3fs
    path = s3_path.replace("s3://", "")
    if not path.endswith("/"):
        path += "/"

    try:
        # Attempt to list objects under the prefix
        contents = fs.ls(path, detail=False)
        return len(contents) > 0
    except Exception:
        # Any failure (missing prefix, permissions, transient S3 errors) is treated as non-existence
        return False

# ------------------ 
# CHECK PATCH FILES
# ------------------
def check_patch_files(row_dict):
    """Check existence of all 3 file types (S1, S2, reference) for a patch.

    Why: Each patch requires three independent S3 prefix checks. 
    Running these concurrently reduces latency caused by S3 network requests.

    Parameters:
        row_dict : (dict) Metadata record for a single patch, including S3 paths.

    Returns:(dict) Existence summary for the patch.
    """
    fs = s3fs.S3FileSystem()

    # Map required data components to their respective S3 paths
    paths = {
        "s1": row_dict["s1_path"],
        "s2": row_dict["s2_path"],
        "reference": row_dict["reference_path"],
    }

    results = {}

    # Run S3 existence checks concurrently within a single patch
    with ThreadPoolExecutor(max_workers=3) as executor:
        future_to_key = {
            executor.submit(check_s3_folder_exists, fs, path): key
            for key, path in paths.items()
        }
        for future in as_completed(future_to_key):
            key = future_to_key[future]
            try:
                results[key] = future.result()
            except Exception:
                # Defensive fallback in case any check fails unexpectedly
                results[key] = False

    # Aggregate results for downstream analysis and reporting
    return {
        "patch_id": row_dict["patch_id"],
        "all_exist": all(results.values()),
        "s1_exists": results["s1"],
        "s2_exists": results["s2"],
        "ref_exists": results["reference"],
    }

# ------------------ 
# SAMPLE STRATIFIED
# ------------------
def sample_stratified(df, fraction):
    """Sample data while maintaining the same split distribution (train/val/test ratios).

    Why: When validating only a subset of the dataset, stratified sampling ensures
    that each split remains proportionally represented. This avoids biased
    validation results, especially for small sample fractions.

    Parameters:
        df : (pandas.DataFrame) Metadata table.
        fraction : (float) Fraction of data to sample.

    Returns: (pandas.DataFrame) Sampled metadata table.
    """
    if fraction >= 1.0:
        return df
    if "split" not in df.columns:
        # Fall back to random sampling if split information is unavailable
        return df.sample(frac=fraction, random_state=42)
    sampled = df.groupby("split", group_keys=False).apply(
        lambda x: x.sample(frac=fraction, random_state=42), include_groups=False
    )
    return sampled.reset_index(drop=True)

# ------------------ 
# CHECK FULL FILES
# ------------------
def check_files(metadata_path, output_path, fraction=1.0, workers=50):
    """
    Orchestrates the full validation workflow.

    Steps:
    ------
    1. Load patch metadata from Parquet
    2. Optionally sample patches (stratified by split)
    3. Check S3 file existence in parallel across patches
    4. Write a JSON summary of results

    Parameters:
        metadata_path : (str) Path to Parquet metadata file (S3 or local).
        output_path : (str) Path to output JSON (S3 or local).
        fraction : (float) Fraction of dataset to validate.
        workers : (int) Number of parallel workers for patch-level checks.

    Returns: None
    """
    print(f"Reading metadata from {metadata_path}")
    table = pq.read_table(metadata_path)
    df = table.to_pandas()
    print(f"Total patches: {len(df)}")

    # Optionally reduce dataset size for faster validation
    if fraction < 1.0:
        df = sample_stratified(df, fraction)
        print(f"Sampled {len(df)} patches ({fraction*100:.1f}% stratified by split)")

    patches = df.to_dict("records")
    results_list = []

    # Parallelize patch-level validation across the dataset
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(check_patch_files, patch) for patch in patches]
        for future in tqdm(
            as_completed(futures), total=len(futures), desc="Checking files"
        ):
            try:
                results_list.append(future.result())
            except Exception as e:
                # Log and continue to avoid failing the entire run
                print(f"Error: {e}")

    # Summarize validation results
    all_found = sum(1 for r in results_list if r["all_exist"])
    not_found = len(results_list) - all_found
    missing_patches = [r for r in results_list if not r["all_exist"]]

    results = {
        "all_files_found": all_found,
        "not_found": not_found,
        "missing_patches": missing_patches,
        "total_checked": len(results_list),
        "fraction_used": fraction,
    }

    json_output = json.dumps(results, indent=2)

    # Write output either to S3 or to local filesystem
    if output_path.startswith("s3://"):
        fs = s3fs.S3FileSystem()
        with fs.open(output_path, "w") as f:
            f.write(json_output)
        print(f"Results written to {output_path}")
    else:
        with open(output_path, "w") as f:
            f.write(json_output)
        print(f"Results written to {output_path}")

    print(f"\nSummary: {all_found}/{len(results_list)} patches have all files")


# ------------------
# MAIN ARGUMENT PARSER
# ------------------
def main():
    """
    Parse command-line arguments and launch the S3 validation process.

    Command-line arguments:
        --meta : (str) S3 path to metadata parquet
        --out : (str) S3 or local path for output JSON
        --frac : (float) Fraction of dataset to validate
        --workers : (int) Number of parallel workers

    """
    parser = argparse.ArgumentParser(description="Check BigEarthNet files on S3 with parallel requests")
    parser.add_argument("--meta", default="s3://ubs-homes/erasmus/raj/dlproject/metadata_with_paths.parquet", 
                        help="S3 path to metadata parquet",) # metadata_with_paths.parquet
    parser.add_argument("--out",default="s3://ubs-homes/erasmus/raj/dlproject/check_s3/validation.json", 
                        help="S3 or local path for output JSON",) # validation.json
    parser.add_argument( "--frac", type=float, default=0.001, 
                        help="Fraction (0.0-1.0), stratified by split",) # fraction
    parser.add_argument("--workers", type=int, default=50, 
                        help="Parallel workers") # number of workers
    args = parser.parse_args()
    if not 0 < args.frac <= 1.0:
        raise ValueError("frac must be between 0.0 and 1.0")
    check_files(args.meta, args.out, args.frac, args.workers)

if __name__ == "__main__":
    main()
