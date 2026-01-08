"""
BigEarthNet S3 File Existence Checker

Validates that all BigEarthNet dataset files (S1, S2, and reference maps) exist on S3.
Reads metadata parquet, checks each patch's S1/S2/reference paths, and outputs results as JSON.

Usage:
    python scripts/check_s3_files.py \
        --metadata-path s3://bucket/path/metadata.parquet \
        --output-path s3://bucket/path/results.json

Output JSON contains:
    - all_files_found: count of patches with all files present
    - not_found: count of patches with missing files
    - missing_patches: list of patches with missing files and which ones are missing
    - total_checked: total patches checked
"""

import argparse
import json
import boto3
import pyarrow.parquet as pq
from tqdm import tqdm

def check_s3_folder_exists(s3_client, s3_path):
    """Check if S3 folder exists by listing objects with the prefix"""
    if not s3_path or not isinstance(s3_path, str) or not s3_path.startswith('s3://'):
        return False
    
    path_parts = s3_path.replace('s3://', '').split('/', 1)
    bucket = path_parts[0]
    prefix = path_parts[1] if len(path_parts) > 1 else ''
    
    if not prefix.endswith('/'):
        prefix += '/'
    
    response = s3_client.list_objects_v2(Bucket=bucket, Prefix=prefix, MaxKeys=1)
    return 'Contents' in response

def main(metadata_path, output_path):
    """Main function to check all patches and write results"""
    s3_client = boto3.client('s3')
    
    # Read metadata from S3
    table = pq.read_table(metadata_path)
    df = table.to_pandas()
    
    # Check each patch's S1, S2, and reference files
    # Initialize results tracking
    results = {'all_files_found': 0, 'not_found': 0, 'missing_patches': []}
    
    for idx in tqdm(range(len(df)), desc="Checking files"):
        row = df.iloc[idx]
        
        s1_exists = check_s3_folder_exists(s3_client, row['s1_path'])
        s2_exists = check_s3_folder_exists(s3_client, row['s2_path'])
        ref_exists = check_s3_folder_exists(s3_client, row['reference_path'])
        
        if s1_exists and s2_exists and ref_exists:
            results['all_files_found'] += 1
        else:
            results['not_found'] += 1
            results['missing_patches'].append({
                'patch_id': row['patch_id'],
                's1_exists': s1_exists,
                's2_exists': s2_exists,
                'ref_exists': ref_exists
            })
    
    results['total_checked'] = len(df)
    # Write results to S3 or local file
    
    json_output = json.dumps(results, indent=2)
    
    if output_path.startswith('s3://'):
        bucket, key = output_path.replace('s3://', '').split('/', 1)
        s3_client.put_object(Bucket=bucket, Key=key, Body=json_output, ContentType='application/json')
        print(f"Results written to {output_path}")
    else:
        with open(output_path, 'w') as f:
            f.write(json_output)
        print(f"Results written to {output_path}")
    
    print(f"\nSummary: {results['all_files_found']}/{results['total_checked']} patches have all files")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Check if all BigEarthNet files exist on S3')
    parser.add_argument('--metadata-path', required=True, help='S3 path to metadata parquet file')
    parser.add_argument('--output-path', required=True, help='S3 or local path for output JSON results')
    
    args = parser.parse_args()
    main(args.metadata_path, args.output_path)