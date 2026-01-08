"""Convert BigEarthNet TIF files to Parquet format for distributed training."""

import argparse
import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np
import boto3
from rasterio.io import MemoryFile
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from tqdm import tqdm

def read_s3_tif(s3_path):
    """Read TIF file from S3 directly into memory"""
    s3_client = boto3.client('s3')
    bucket, key = s3_path.replace('s3://', '').split('/', 1)
    obj = s3_client.get_object(Bucket=bucket, Key=key)
    with MemoryFile(obj['Body'].read()) as memfile:
        with memfile.open() as dataset:
            return dataset.read()

def pad_to_size(array, target_shape, pad_value=0):
    """Pad array to target shape with specified value"""
    pad_width = [(0, max(0, target - current)) 
                 for target, current in zip(target_shape, array.shape)]
    return np.pad(array, pad_width, mode='constant', constant_values=pad_value)

def process_patch(row_dict, target_size=(120, 120)):
    try:
        s2_bands = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B11', 'B12']
        s3_paths = {'s1_vv': f"{row_dict['s1_path']}/{row_dict['s1_name']}_VV.tif", 's1_vh': f"{row_dict['s1_path']}/{row_dict['s1_name']}_VH.tif", 'reference': f"{row_dict['reference_path']}/{row_dict['patch_id']}_reference_map.tif"}
        for band in s2_bands:
            s3_paths[f's2_{band}'] = f"{row_dict['s2_path']}/{row_dict['patch_id']}_{band}.tif"
        
        # Parallel download (15 concurrent S3 requests)
        file_data = {}
        with ThreadPoolExecutor(max_workers=15) as executor:
            future_to_key = {executor.submit(read_s3_tif, path): key for key, path in s3_paths.items()}
            for future in as_completed(future_to_key):
                key = future_to_key[future]
                try:
                    file_data[key] = future.result()[0]
                except Exception as e:
                    raise Exception(f"Failed {key}: {e}")
        
        s1_vv = pad_to_size(file_data['s1_vv'], target_size)
        s1_vh = pad_to_size(file_data['s1_vh'], target_size)
        s1_data = np.stack([s1_vv, s1_vh], axis=-1).astype(np.float32)
        
        # Process S2 data
        s2_band_arrays = []
        for band in s2_bands:
            band_data = pad_to_size(file_data[f's2_{band}'], target_size)
            s2_band_arrays.append(band_data)
        s2_data = np.stack(s2_band_arrays, axis=-1).astype(np.float32)
        
        # Process reference map
        reference = pad_to_size(file_data['reference'], target_size)
        reference = np.expand_dims(reference, axis=-1).astype(np.uint8)
        
        return {
            'patch_id': row_dict['patch_id'],
            's1_data': s1_data.tobytes(),
            's2_data': s2_data.tobytes(),
            'reference': reference.tobytes(),
            'labels': ','.join(row_dict['labels']),
            's1_shape': s1_data.shape,
            's2_shape': s2_data.shape,
            'ref_shape': reference.shape,
        }
    except Exception as e:
        print(f"Error processing {row_dict['patch_id']}: {e}")
        return None

def sample_stratified(df, fraction):
    if fraction >= 1.0:
        return df
    if 'split' not in df.columns:
        return df.sample(frac=fraction, random_state=42)
    sampled = df.groupby('split', group_keys=False).apply(lambda x: x.sample(frac=fraction, random_state=42), include_groups=False)
    return sampled.reset_index(drop=True)

def convert_files(metadata_path, output_path, fraction=1.0, workers=10, batch_size=100):
    # Read metadata
    print(f"Reading metadata from {metadata_path}")
    table = pq.read_table(metadata_path)
    df = table.to_pandas()
    
    print(f"Total patches in metadata: {len(df)}")
    
    # Apply stratified sampling
    if fraction < 1.0:
        df = sample_stratified(df, fraction)
        print(f"Sampled {len(df)} patches ({fraction*100:.1f}% stratified by split)")
    
    print(f"Processing {len(df)} patches with {workers} workers")
    
    # Define PyArrow schema
    schema = pa.schema([
        ('patch_id', pa.string()),
        ('s1_data', pa.binary()),
        ('s2_data', pa.binary()),
        ('reference', pa.binary()),
        ('labels', pa.string()),
        ('s1_shape', pa.list_(pa.int32())),
        ('s2_shape', pa.list_(pa.int32())),
        ('ref_shape', pa.list_(pa.int32())),
    ])
    
    # Process in batches and write incrementally
    s3_client = boto3.client('s3')
    bucket, key_prefix = output_path.replace('s3://', '').split('/', 1)
    
    batch_num = 0
    records = df.to_dict('records')
    
    for i in tqdm(range(0, len(records), batch_size), desc="Processing batches"):
        batch = records[i:i+batch_size]
        
        # Process batch in parallel
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(process_patch, row) for row in batch]
            results = [f.result() for f in as_completed(futures) if f.result() is not None]
        
        if not results:
            continue
            
        # Create PyArrow table from batch
        batch_table = pa.table({
            'patch_id': [r['patch_id'] for r in results],
            's1_data': [r['s1_data'] for r in results],
            's2_data': [r['s2_data'] for r in results],
            'reference': [r['reference'] for r in results],
            'labels': [r['labels'] for r in results],
            's1_shape': [list(r['s1_shape']) for r in results],
            's2_shape': [list(r['s2_shape']) for r in results],
            'ref_shape': [list(r['ref_shape']) for r in results],
        }, schema=schema)
        
        # Write to S3
        output_key = f"{key_prefix}/part-{batch_num:05d}.parquet"
        pq.write_table(batch_table, f"s3://{bucket}/{output_key}")
        
        print(f"Wrote batch {batch_num} with {len(results)} patches")
        batch_num += 1
    
    print(f"\nDataset saved to {output_path}")
    print(f"Total batches: {batch_num}")
    print(f"Fraction processed: {fraction*100:.1f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert BigEarthNet to Parquet format')
    parser.add_argument('--meta', required=True, help='S3 path to metadata parquet')
    parser.add_argument('--out', required=True, help='S3 path for output Parquet data')
    parser.add_argument('--frac', type=float, default=1.0, help='Fraction of data (0.0-1.0), stratified by split')
    parser.add_argument('--workers', type=int, default=10, help='Parallel workers')
    parser.add_argument('--batch', type=int, default=100, help='Batch size')
    
    args = parser.parse_args()
    
    if not 0 < args.frac <= 1.0:
        raise ValueError("frac must be between 0.0 and 1.0")
    convert_files(args.meta, args.out, args.frac, args.workers, args.batch)

def main():
    parser = argparse.ArgumentParser(description='Convert BigEarthNet to Parquet format')
    parser.add_argument('--meta', required=True, help='S3 path to metadata parquet')
    parser.add_argument('--out', required=True, help='S3 path for output Parquet data')
    parser.add_argument('--frac', type=float, default=1.0, help='Fraction (0.0-1.0), stratified by split')
    parser.add_argument('--workers', type=int, default=10, help='Parallel workers')
    parser.add_argument('--batch', type=int, default=100, help='Batch size')
    args = parser.parse_args()
    if not 0 < args.frac <= 1.0:
        raise ValueError("frac must be between 0.0 and 1.0")
    convert_files(args.meta, args.out, args.frac, args.workers, args.batch)

if __name__ == "__main__":
    main()