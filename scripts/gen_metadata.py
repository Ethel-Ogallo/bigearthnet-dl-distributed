"""Add S3 paths (s1_path, s2_path, reference_path) to BigEarthNet metadata."""

import argparse
import pyarrow.parquet as pq
import pyarrow.compute as pc

def generate_metadata_with_paths(input_path, output_path):
    print(f"Reading metadata from {input_path}")
    table = pq.read_table(input_path)
    
    # Extract folder prefix from s1_name (first 5 fields)
    s1_prefix = pc.replace_substring_regex(table['s1_name'], r'((?:[^_]+_){4}[^_]+)_.*', r'\1')
    s1_path = pc.binary_join_element_wise("s3://ubs-datasets/bigearthnet/BigEarthNet-S1/", s1_prefix, '')
    s1_path = pc.binary_join_element_wise(s1_path, table['s1_name'], '/')
    
    # Extract folder prefix from patch_id (first 6 fields)
    s2_prefix = pc.replace_substring_regex(table['patch_id'], r'((?:[^_]+_){5}[^_]+)_.*', r'\1')
    s2_path = pc.binary_join_element_wise("s3://ubs-datasets/bigearthnet/BigEarthNet-S2/", s2_prefix,'')
    s2_path = pc.binary_join_element_wise(s2_path, table['patch_id'], '/')
    
    ref_path = pc.binary_join_element_wise("s3://ubs-datasets/bigearthnet/Reference_Maps/", s2_prefix, '')
    ref_path = pc.binary_join_element_wise(ref_path, table['patch_id'], '/')
    
    table = table.append_column('s1_path', s1_path)
    table = table.append_column('s2_path', s2_path)
    table = table.append_column('reference_path', ref_path)
    
    print(f"Writing enhanced metadata to {output_path}")
    pq.write_table(table, output_path)
    print(f"Done! {len(table)} rows with s1_path, s2_path, reference_path")

def main():
    parser = argparse.ArgumentParser(description='Generate BigEarthNet metadata with S3 paths')
    parser.add_argument('--meta', required=True, help='S3 path to input metadata parquet')
    parser.add_argument('--out', required=True, help='S3 path for output metadata with paths')
    args = parser.parse_args()
    generate_metadata_with_paths(args.meta, args.out)

if __name__ == "__main__":
    main()
