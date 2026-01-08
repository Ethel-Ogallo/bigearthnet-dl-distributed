"""Check BigEarthNet S3 file existence with parallel requests."""

import argparse
import json
import boto3
import pyarrow.parquet as pq
from concurrent.futures import ThreadPoolExecutor, as_completed
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

def check_patch_files(row_dict):
    s3_client = boto3.client('s3')
    paths = {'s1': row_dict['s1_path'], 's2': row_dict['s2_path'], 'reference': row_dict['reference_path']}
    results = {}
    with ThreadPoolExecutor(max_workers=3) as executor:
        future_to_key = {executor.submit(check_s3_folder_exists, s3_client, path): key for key, path in paths.items()}
        for future in as_completed(future_to_key):
            key = future_to_key[future]
            try:
                results[key] = future.result()
            except Exception:
                results[key] = False
    return {'patch_id': row_dict['patch_id'], 'all_exist': all(results.values()), 's1_exists': results['s1'], 's2_exists': results['s2'], 'ref_exists': results['reference']}

def sample_stratified(df, fraction):
    if fraction >= 1.0:
        return df
    if 'split' not in df.columns:
        return df.sample(frac=fraction, random_state=42)
    sampled = df.groupby('split', group_keys=False).apply(lambda x: x.sample(frac=fraction, random_state=42), include_groups=False)
    return sampled.reset_index(drop=True)

def check_files(metadata_path, output_path, fraction=1.0, workers=50):
    print(f"Reading metadata from {metadata_path}")
    table = pq.read_table(metadata_path)
    df = table.to_pandas()
    print(f"Total patches: {len(df)}")
    
    if fraction < 1.0:
        df = sample_stratified(df, fraction)
        print(f"Sampled {len(df)} patches ({fraction*100:.1f}% stratified by split)")
    
    patches = df.to_dict('records')
    results_list = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(check_patch_files, patch) for patch in patches]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Checking files"):
            try:
                results_list.append(future.result())
            except Exception as e:
                print(f"Error: {e}")
    
    # Aggregate results
    all_found = sum(1 for r in results_list if r['all_exist'])
    not_found = len(results_list) - all_found
    missing_patches = [r for r in results_list if not r['all_exist']]
    
    results = {
        'all_files_found': all_found,
        'not_found': not_found,
        'missing_patches': missing_patches,
        'total_checked': len(results_list),
        'fraction_used': fraction
    }
    results = {
        'all_files_found': all_found,
        'not_found': not_found,
        'missing_patches': missing_patches,
        'total_checked': len(results_list),
        'fraction_used': fraction
    }
    
    # Write results to S3 or local file
    json_output = json.dumps(results, indent=2)
    
    if output_path.startswith('s3://'):
        s3_client = boto3.client('s3')
        bucket, key = output_path.replace('s3://', '').split('/', 1)
        s3_client.put_object(Bucket=bucket, Key=key, Body=json_output, ContentType='application/json')
        print(f"Results written to {output_path}")
    else:
        with open(output_path, 'w') as f:
            f.write(json_output)
        print(f"Results written to {output_path}")
    
    print(f"\nSummary: {all_found}/{len(results_list)} patches have all files")

def main():
    parser = argparse.ArgumentParser(description='Check BigEarthNet files on S3 with parallel requests')
    parser.add_argument('--meta', required=True, help='S3 path to metadata parquet')
    parser.add_argument('--out', required=True, help='S3 or local path for output JSON')
    parser.add_argument('--frac', type=float, default=1.0, help='Fraction (0.0-1.0), stratified by split')
    parser.add_argument('--workers', type=int, default=50, help='Parallel workers')
    args = parser.parse_args()
    if not 0 < args.frac <= 1.0:
        raise ValueError("frac must be between 0.0 and 1.0")
    check_files(args.meta, args.out, args.frac, args.workers)

if __name__ == "__main__":
    main()