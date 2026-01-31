"""
Add S3 paths (s1_path, s2_path, reference_path) to BigEarthNet metadata.
Authors: Kshitij Raj & Ethel Ogallo
Last Updated: 30-01-2026


Description:
-------------
Adds S3 paths (s1_path, s2_path, reference_path) to BigEarthNet metadata.
Original metadata contains only patch IDs and filenames. This script pre-computes
the full S3 paths to avoid string concatenation during training, improving
data loading performance.

"""

# ------------------ 
# IMPORTS
# ------------------
import argparse
import pyarrow.compute as pc
import pyarrow.parquet as pq


# ------------------ 
# GENERATE METADATA WITH PATHS
# ------------------
def generate_metadata_with_paths(input_path, output_path):
    """
    Add S3 file paths to BigEarthNet metadata for efficient data access.
    Original metadata only contains patch IDs. Precomputing full S3 paths reduces
    runtime string operations during training and downstream processing.

    Parameters:
        input_path : Path to original metadata Parquet (S3 or local)
        output_path : Path to write enhanced metadata Parquet (S3 or local)

    Returns: None
    """
    print(f"Reading metadata from {input_path}")
    table = pq.read_table(input_path)

    # Extract S1 folder prefix (first 5 underscore-separated fields from s1_name)
    # Example: S1A_IW_GRDH_1SDV_20170613T165043_33UUP_33_62 -> S1A_IW_GRDH_1SDV_20170613T165043
    s1_prefix = pc.replace_substring_regex(
        table["s1_name"], r"((?:[^_]+_){4}[^_]+)_.*", r"\1"
    )

    # Concatenate base S3 path + folder prefix
    s1_path = pc.binary_join_element_wise(
        "s3://ubs-datasets/bigearthnet/BigEarthNet-S1/", s1_prefix, ""
    )
    # Append filename to form full S3 path
    s1_path = pc.binary_join_element_wise(s1_path, table["s1_name"], "/")

    # Extract S2 folder prefix (first 6 underscore-separated fields from patch_id)
    # Example: S2A_MSIL2A_20170613T101031_33_62_patch123 -> S2A_MSIL2A_20170613T101031_33_62
    s2_prefix = pc.replace_substring_regex(
        table["patch_id"], r"((?:[^_]+_){5}[^_]+)_.*", r"\1"
    )

    # Concatenate base S3 path + folder prefix + patch ID to get full S2 path
    s2_path = pc.binary_join_element_wise(
        "s3://ubs-datasets/bigearthnet/BigEarthNet-S2/", s2_prefix, ""
    )
    s2_path = pc.binary_join_element_wise(s2_path, table["patch_id"], "/")

    # Build reference map path similarly
    ref_path = pc.binary_join_element_wise(
        "s3://ubs-datasets/bigearthnet/Reference_Maps/", s2_prefix, ""
    )
    ref_path = pc.binary_join_element_wise(ref_path, table["patch_id"], "/")

    # Append new columns to metadata table
    table = table.append_column("s1_path", s1_path)
    table = table.append_column("s2_path", s2_path)
    table = table.append_column("reference_path", ref_path)

    print(f"Writing enhanced metadata to {output_path}")
    pq.write_table(table, output_path)
    print(f"Done! {len(table)} rows with s1_path, s2_path, reference_path")

# ------------------ 
# MAIN ARGUMENT PARSER
# ------------------
def main():
    """
    Parse command-line arguments and run metadata enhancement.

    Command-line arguments:
        --meta : (str) Input metadata Parquet path (S3 or local)
        --out : (str) Output Parquet path with S3 paths added

    """
    parser = argparse.ArgumentParser(description="Generate BigEarthNet metadata with S3 paths") 
    parser.add_argument("--meta",default="s3://ubs-datasets/bigearthnet/metadata.parquet",
                        help="S3 path to input metadata parquet",) # metadata.parquet input
    parser.add_argument( "--out", default="s3://ubs-homes/erasmus/raj/dlproject/metadata_with_paths.parquet",
                        help="S3 path for output metadata with paths",) # metadata_with_paths.parquet output
    args = parser.parse_args()
    generate_metadata_with_paths(args.meta, args.out)


if __name__ == "__main__":
    main()
