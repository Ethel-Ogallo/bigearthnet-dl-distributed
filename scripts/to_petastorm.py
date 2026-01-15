import argparse
import os
import numpy as np
# import boto3
from rasterio.io import MemoryFile
import pyarrow.parquet as pq
from pyspark.sql import SparkSession
from pyspark.sql.types import IntegerType
from petastorm.etl.dataset_metadata import materialize_dataset
from petastorm.unischema import Unischema, UnischemaField, dict_to_spark_row
from petastorm.codecs import NdarrayCodec, ScalarCodec
import s3fs 
import time 
# # Read S3 GeoTIFF
# def read_s3_tif(s3_path):
#     """Download and read GeoTIFF from S3."""
#     s3_client = boto3.client('s3')
#     bucket, key = s3_path.replace('s3://', '').split('/', 1)
#     obj = s3_client.get_object(Bucket=bucket, Key=key)
#     with MemoryFile(obj['Body'].read()) as memfile:
#         with memfile.open() as dataset:
#             return dataset.read()



def read_s3_tif(s3_path):
    """Download and read GeoTIFF from S3 using s3fs."""
    s3fs = s3fs.S3FileSystem()
    with s3fs.open(s3_path, 'rb') as f:
        with MemoryFile(f) as memfile:
            with memfile.open() as dataset:
                return dataset.read()
    

# Process a single patch
def process_patch_stream(row_dict):
    """Download and combine S1, S2, and label data for a single patch."""
    try:
        s2_bands = ['B02', 'B03', 'B04', 'B08']
        s3_paths = {
            's1_vv': f"{row_dict['s1_path']}/{row_dict['s1_name']}_VV.tif",
            's1_vh': f"{row_dict['s1_path']}/{row_dict['s1_name']}_VH.tif",
            'label': f"{row_dict['reference_path']}/{row_dict['patch_id']}_reference_map.tif"
        }
        for band in s2_bands:
            s3_paths[f's2_{band}'] = f"{row_dict['s2_path']}/{row_dict['patch_id']}_{band}.tif"

        # serially download files
        file_data = {}
        for key, path in s3_paths.items():
            file_data[key] = read_s3_tif(path)[0]

        # Stack S1 + S2
        s1_data = np.stack([file_data['s1_vv'], file_data['s1_vh']], axis=-1).astype(np.float32)
        s2_data = np.stack([file_data[f's2_{band}'] for band in s2_bands], axis=-1).astype(np.float32)
        input_data = np.concatenate([s1_data, s2_data], axis=-1).astype(np.float32)

        label = file_data['label'].astype(np.uint8)

        return {
            'patch_id': row_dict['patch_id'],
            'input_data': input_data,
            'label': label
        }

    except Exception as e:
        print(f"Error processing {row_dict['patch_id']}: {e}")
        return None

# Split and sample DataFrame
def split_and_sample(df, fraction=1.0):
    """Split DataFrame into train, val, test and sample fraction from each split."""
    splits = {}
    for split_name in ['train', 'validation', 'test']:
        split_df = df[df['split'] == split_name]
        if fraction < 1.0:
            split_df = split_df.sample(frac=fraction, random_state=42)
        splits[split_name] = split_df.reset_index(drop=True)
    return splits['train'], splits['validation'], splits['test']


# Convert files to petastorm
def convert_to_petastorm(metadata_path, output_dir, fraction=1.0, target_size=(120, 120),
                         workers=5, executor_mem='8g', driver_mem='4g', core=4, n_executor=3):
    
    """Convert patches into Petastorm datasets using streaming generator."""
    print(f"Reading metadata from {metadata_path}")
    table = pq.read_table(metadata_path)
    df = table.to_pandas()
    print(f"Total patches: {len(df)}")

    train_df, val_df, test_df = split_and_sample(df, fraction)
    datasets = {'train': train_df, 'validation': val_df, 'test': test_df}

    all_patch_ids = df['patch_id'].unique()
    patch_id_to_int = {pid: i for i, pid in enumerate(all_patch_ids)}

    # Start Spark session
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
    output_paths = {}

    input_shape = (*target_size, 6)
    label_shape = target_size

    # Petastorm schema
    InputSchema = Unischema('InputSchema', [
        UnischemaField('patch_id_int', np.int32, (), ScalarCodec(IntegerType()), False),
        UnischemaField('input_data', np.float32, input_shape, NdarrayCodec(), False),
        UnischemaField('label', np.uint8, label_shape, NdarrayCodec(), False),
    ])

    try:
        for split_name, split_df in datasets.items():
            if split_df.empty:
                continue

            print(f"\nProcessing {split_name} split ({len(split_df)} patches)...")

            def row_generator(index):
                row = split_df.iloc[index]
                patch = process_patch_stream(row)
                if patch:
                    return dict_to_spark_row(
                        InputSchema,
                        {
                            'patch_id_int': patch_id_to_int[patch['patch_id']],
                            'input_data': patch['input_data'],
                            'label': patch['label']
                        }
                    )
                return None

            split_path = os.path.join(output_dir, split_name)
            if not split_path.startswith(("s3://", "s3a://")):
                os.makedirs(split_path, exist_ok=True)

            # Streaming generator into Petastorm 
            with materialize_dataset(spark, split_path, InputSchema) as writer:
                rows_rdd = (
                    sc.parallelize(range(len(split_df)))
                      .map(row_generator)
                      .filter(lambda x: x is not None)
                )
                rows_df = spark.createDataFrame(rows_rdd, InputSchema.as_spark_schema())
                rows_df.write.mode("overwrite").parquet(split_path)

            output_paths[split_name] = split_path
            print(f"{split_name} dataset saved: {split_path}")

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
    start_time = time.time()
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
    end_time = time.time()
    print(f"Conversion completed in {end_time - start_time:.2f} seconds.")

if __name__ == "__main__":
    main()



