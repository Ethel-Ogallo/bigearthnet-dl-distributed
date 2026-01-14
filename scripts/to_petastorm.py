import argparse
import os
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import boto3
from rasterio.io import MemoryFile
import pyarrow.parquet as pq
from pyspark.sql import SparkSession
from pyspark.sql.types import StringType, IntegerType
from petastorm.etl.dataset_metadata import materialize_dataset
from petastorm.unischema import Unischema, UnischemaField, dict_to_spark_row
from petastorm.codecs import NdarrayCodec, ScalarCodec



def read_s3_tif(s3_path):
    """Download and read GeoTIFF from S3."""
    s3_client = boto3.client('s3')
    bucket, key = s3_path.replace('s3://', '').split('/', 1)
    obj = s3_client.get_object(Bucket=bucket, Key=key)
    with MemoryFile(obj['Body'].read()) as memfile:
        with memfile.open() as dataset:
            return dataset.read()

def pad_to_size(array, target_shape, pad_value=0):
    """Pad 2D array to target shape with zeros."""
    pad_width = [(0, max(0, target - current)) 
                 for target, current in zip(target_shape, array.shape)]
    return np.pad(array, pad_width, mode='constant', constant_values=pad_value)

def process_patch(row_dict, target_size=(120, 120)):
    """Download, process, and combine S1, S2, and label data for a single patch."""
    try:

        # Build S3 paths for all bands
        # s2_bands = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B11', 'B12']
        s2_bands = [ 'B02', 'B03', 'B04', 'B08']
        s3_paths = {
            's1_vv': f"{row_dict['s1_path']}/{row_dict['s1_name']}_VV.tif",
            's1_vh': f"{row_dict['s1_path']}/{row_dict['s1_name']}_VH.tif",
            'label': f"{row_dict['reference_path']}/{row_dict['patch_id']}_reference_map.tif"

        }
        for band in s2_bands:
            s3_paths[f's2_{band}'] = f"{row_dict['s2_path']}/{row_dict['patch_id']}_{band}.tif"
        
        # Download all files in parallel (15 concurrent requests)
        file_data = {}
        with ThreadPoolExecutor(max_workers=15) as executor:
            future_to_key = {executor.submit(read_s3_tif, path): key for key, path in s3_paths.items()}
            for future in as_completed(future_to_key):
                key = future_to_key[future]
                file_data[key] = future.result()[0]  # Get first band
        
        # Stack S1 bands (VV, VH) -> (120, 120, 2)
        s1_data = np.stack([
            pad_to_size(file_data['s1_vv'], target_size),
            pad_to_size(file_data['s1_vh'], target_size)
        ], axis=-1).astype(np.float32)
        
        # Stack S2 bands (4 bands) -> (120, 120, 4)
        s2_data = np.stack([
            pad_to_size(file_data[f's2_{band}'], target_size) for band in s2_bands
        ], axis=-1).astype(np.float32)
        
        # combine s1+s2 -> (120, 120, 6)
        input_data = np.concatenate([s1_data, s2_data], axis=-1).astype(np.float32)

        # Process label map -> (120, 120)
        label = pad_to_size(file_data['label'], target_size).astype(np.uint8)
        
        return {
            'patch_id': row_dict['patch_id'],
            'input_data': input_data,
            'label': label,
            'split': row_dict['split']  
        }
    except Exception as e:
        print(f"Error processing {row_dict['patch_id']}: {e}")
        return None

def sample_stratified(df, fraction):
    """Sample dataset maintaining train/val/test split proportions."""
    if fraction >= 1.0:
        return df
    if 'split' not in df.columns:
        return df.sample(frac=fraction, random_state=42)
    return df.groupby('split', group_keys=False).apply(
        lambda x: x.sample(frac=fraction, random_state=42)
    ).reset_index(drop=True)

def split_dataset(df, fraction=1.0):
    """Split DataFrame into train, val, test sets."""
    df = sample_stratified(df, fraction)
    
    train_df = df[df['split'] == 'train'].reset_index(drop=True)
    val_df   = df[df['split'] == 'val'].reset_index(drop=True)
    test_df  = df[df['split'] == 'test'].reset_index(drop=True)
    
    return train_df, val_df, test_df

def convert_to_petastorm(metadata_path, output_dir, fraction=1.0, target_size=(120, 120),
                         workers=10, executor_mem='8g', driver_mem='4g', core=4, n_executor=3):
    """Convert BigEarthNet patches into Petastorm datasets with integer patch IDs."""
    
    print(f"Reading metadata from {metadata_path}")
    table = pq.read_table(metadata_path)
    df = table.to_pandas()
    print(f"Total patches: {len(df)}")

    # Sample and split dataset
    df_sampled = sample_stratified(df, fraction)
    train_df, val_df, test_df = split_dataset(df_sampled)
    datasets = {'train': train_df, 'val': val_df, 'test': test_df}

    # Map patch IDs to integers
    all_patch_ids = df_sampled['patch_id'].unique()
    patch_id_to_int = {pid: i for i, pid in enumerate(all_patch_ids)}

    input_shape = (*target_size, 6)
    label_shape = target_size

    # Define Petastorm schema
    InputSchema = Unischema('InputSchema', [
        UnischemaField('patch_id_int', np.int32, (), ScalarCodec(IntegerType()), False), # as integer because petastorm does not support string keys
        UnischemaField('input_data', np.float32, input_shape, NdarrayCodec(), False),
        UnischemaField('label', np.uint8, label_shape, NdarrayCodec(), False),
    ])

    output_paths = {}

    # Spark session required for Petastorm
    spark = (
        SparkSession.builder.appName("petastorm_bigearthnet")
        .config("spark.hadoop.fs.s3a.aws.credentials.provider",
                "com.amazonaws.auth.DefaultAWSCredentialsProviderChain")
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
        .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:3.3.1")
        .config('spark.executor.memory', executor_mem)
        .config('spark.driver.memory', driver_mem)
        .config('spark.executor.instances', n_executor)
        .config('spark.executor.cores', core)
        .getOrCreate()
    ) 

    sc = spark.sparkContext

    try:
        for split_name, split_df in datasets.items():
            if split_df.empty:
                print(f"No patches in {split_name}. Skipping.")
                continue

            print(f"\nProcessing {split_name} split ({len(split_df)} patches)...")
            records = split_df.to_dict('records')
            processed_patches = []

            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = [executor.submit(process_patch, row, target_size) for row in records]
                for f in tqdm(as_completed(futures), total=len(futures), desc=f"Processing {split_name}"):
                    res = f.result()
                    if res:
                        processed_patches.append(res)

            def row_generator(data):
                for item in data:
                    yield {
                        'patch_id_int': patch_id_to_int[item['patch_id']], # map to integer ID
                        'input_data': item['input_data'],
                        'label': item['label'],
                    }

            split_path = os.path.join(output_dir, split_name)
            if not split_path.startswith(("s3://", "s3a://")):
                os.makedirs(split_path, exist_ok=True)

            print(f"Materializing {split_name} dataset at {split_path}...")

            with materialize_dataset(spark, split_path, InputSchema) as writer:
                rows_rdd = sc.parallelize(list(row_generator(processed_patches))) \
                             .map(lambda x: dict_to_spark_row(InputSchema, x))
                rows_df = spark.createDataFrame(rows_rdd, InputSchema.as_spark_schema())
                rows_df.coalesce(4).write.mode("overwrite").parquet(split_path)

            output_paths[split_name] = split_path
            print(f"{split_name} dataset saved: {split_path} ({len(processed_patches)} patches)")

    finally:
        spark.stop()
        print("Spark session stopped.")

    return output_paths


# ---- CLI ----
def main():
    import argparse
    parser = argparse.ArgumentParser(description="Convert BigEarthNet patches to Petastorm format")
    parser.add_argument("--meta", type=str, required=True, help="Metadata Parquet path")
    parser.add_argument("--out", type=str, required=True, help="Output Petastorm dataset dir (S3 or local)")
    parser.add_argument("--frac", type=float, default=1.0, help="Fraction of dataset to sample")
    parser.add_argument("--workers", type=int, default=10, help="Number of parallel threads")
    parser.add_argument("--target_size", type=int, nargs=2, default=[120, 120], help="Target patch size H W")
    parser.add_argument("--executor-mem", required=False, help="executor memory", default='8g')
    parser.add_argument("--driver-mem",required=False, help="driver memory", default='4g')
    parser.add_argument("--core", type=int, default=4) 
    parser.add_argument("--n_executor", type=int, default=3)
    args = parser.parse_args()

    convert_to_petastorm(
        metadata_path=args.meta,
        output_dir=args.out,
        fraction=args.frac,
        target_size=tuple(args.target_size),
        workers=args.workers,
        executor_mem=args.executor_mem,
        driver_mem=args.driver_mem,
        core=args.core,
        n_executor=args.n_executor
    )

if __name__ == "__main__":
    main()


