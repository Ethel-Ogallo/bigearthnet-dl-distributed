import os
import time

import numpy as np
import pyarrow. parquet as pq
import s3fs
from petastorm. codecs import NdarrayCodec
from petastorm.etl. dataset_metadata import materialize_dataset
from petastorm.unischema import Unischema, UnischemaField, dict_to_spark_row
from pyspark.sql import SparkSession
from rasterio.io import MemoryFile

# from profiler import Profiler
from scripts.profiler import Profiler


BIGEARTH_IDS = [
    111,
    112,
    121,
    122,
    123,
    124,
    131,
    132,
    133,
    141,
    142,
    211,
    212,
    213,
    221,
    222,
    223,
    231,
    241,
    242,
    243,
    244,
    311,
    312,
    313,
    321,
    322,
    323,
    324,
    331,
    332,
    333,
    334,
    335,
    411,
    412,
    421,
    422,
    423,
    511,
    512,
    521,
    522,
    523,
    999,
]

LABEL_MAPPING = np.zeros(1000, dtype=np.uint8)
for idx, class_id in enumerate(BIGEARTH_IDS):
    LABEL_MAPPING[class_id] = idx


def remap_labels(label_array):
    """Remap BigEarthNet class IDs to sequential indices"""
    return LABEL_MAPPING[label_array]

def read_s3_tif(s3_paths, fs):
    """Read TIF file from S3 and return as numpy array"""
    results = {}
    for key, s3_path in s3_paths.items():
        s3_path = s3_path.replace("s3a://", "s3://")
        with fs.open(s3_path, "rb") as f:
            with MemoryFile(f.read()) as memfile:
                with memfile.open() as dataset:
                    results[key] = dataset.read()[0]
    return results

def process_patch_stream(row_dict, fs):
    """Process a single patch by loading and combining S1, S2, and label data"""
    try:
        # Define S2 bands to load
        s2_bands = ["B02", "B03", "B04", "B08"]

        # Construct S3 paths for all required files
        s3_paths = {
            "s1_vv": f"{row_dict['s1_path']}/{row_dict['s1_name']}_VV.tif",
            "s1_vh": f"{row_dict['s1_path']}/{row_dict['s1_name']}_VH.tif",
            "label": f"{row_dict['reference_path']}/{row_dict['patch_id']}_reference_map.tif",
        }
        for band in s2_bands:
            s3_paths[f"s2_{band}"] = (
                f"{row_dict['s2_path']}/{row_dict['patch_id']}_{band}.tif"
            )

        # Load all TIF files from S3
        file_data = read_s3_tif(s3_paths, fs)
        # Combine S1 data (VV and VH polarizations)
        s1_data = np.stack([file_data["s1_vv"], file_data["s1_vh"]], axis=-1).astype(
            np.float32
        )
        # Combine S2 data (4 bands)
        s2_data = np.stack(
            [file_data[f"s2_{band}"] for band in s2_bands], axis=-1
        ).astype(np.float32)
        # Concatenate S1 and S2 to create final image (120x120x6)
        image = np.concatenate([s1_data, s2_data], axis=-1)

        label = remap_labels(file_data["label"]).astype(np.uint8)
        return {"image": image, "label": label}
    except Exception as e:
        print(f"Error processing {row_dict['patch_id']}: {e}")
        return None


def split_and_sample(df, fraction=1.0):
    """Split dataset into train/validation/test and optionally sample a fraction"""
    splits = {}
    for split_name in ["train", "validation", "test"]:
        split_df = df[df["split"] == split_name]
        # Sample fraction of data if specified
        if fraction < 1.0:
            split_df = split_df.sample(frac=fraction, random_state=42)
        splits[split_name] = split_df.reset_index(drop=True)
    return splits["train"], splits["validation"], splits["test"]


def process_partition(rows, schema):
    """Process partition with SHARED S3 connection"""
    fs = s3fs.S3FileSystem(anon=False, client_kwargs={"region_name":  "eu-west-1"})
    
    for row_dict in rows:
        try:
            patch = process_patch_stream(row_dict, fs)
            yield dict_to_spark_row(schema, {"image": patch["image"], "label": patch["label"]})
        except Exception as e:
            print(f"Error processing {row_dict. get('patch_id', 'unknown')}: {e}")


def calculate_partitions(num_samples, target_file_mb=50, target_size=(120, 120)):
    bytes_per_sample = target_size[0] * target_size[1] * 6 * 4 + target_size[0] * target_size[1]
    estimated_compressed = bytes_per_sample * 0.4
    samples_per_partition = int((target_file_mb * 1024 * 1024) / estimated_compressed)
    num_partitions = max(1, (num_samples + samples_per_partition - 1) // samples_per_partition)
    return num_partitions, samples_per_partition


def convert_to_petastorm(
    metadata_path,
    output_dir,
    fraction=1.0,
    target_size=(120, 120),
    executor_mem="8g",
    driver_mem="4g",
    core=4,
    n_executor=3,
    target_file_mb=50,
    p_name="conversion",
    args_str="",
):
    """Convert BigEarthNet TIF files to Petastorm format using Spark"""
    profiler = Profiler()
    profiler.log(f"Args: {args_str}")

    # Step 1: Read metadata parquet file
    with profiler.step("read_metadata"):
        print(f"Reading metadata from {metadata_path}")
        fs = s3fs.S3FileSystem(anon=False, client_kwargs={"region_name": "eu-west-1"})
        table = pq.read_table(metadata_path.replace("s3a://", "s3://"), filesystem=fs)
        df = table.to_pandas()
        print(f"Total patches: {len(df)}")
    profiler.record("fraction", fraction)
    profiler.record("total_patches", len(df))

    # Step 2: Split data into train/val/test and sample fraction
    with profiler.step("split_and_sample", fraction=fraction):
        train_df, val_df, test_df = split_and_sample(df, fraction)
        datasets = {"train": train_df, "validation": val_df, "test":  test_df}

    profiler.record("train_samples", len(train_df))
    profiler.record("validation_samples", len(val_df))
    profiler.record("test_samples", len(test_df))

    # Step 3: Initialize Spark session with S3 configuration
    with profiler.step(
        "spark_init",
        executor_mem=executor_mem,
        driver_mem=driver_mem,
        cores=core,
        executors=n_executor,
    ):
        spark = (
            SparkSession.builder.appName("petastorm_bigearthnet")
            .config(
                "spark.hadoop.fs.s3a.aws.credentials.provider",
                "com.amazonaws.auth.DefaultAWSCredentialsProviderChain",
            )
            .config(
                "spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem"
            )
            .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:3.3.1")
            .config("spark.executor.memory", executor_mem)
            .config("spark.driver.memory", driver_mem)
            .config("spark.executor.instances", n_executor)
            .config("spark.executor.cores", core)
            .getOrCreate()
        )

    profiler.record("spark_executors", n_executor)
    profiler.record("spark_cores", core)
    profiler.record("spark_executor_memory", executor_mem)
    profiler.record("spark_driver_memory", driver_mem)

    sc = spark.sparkContext
    output_paths = {}
    input_shape = (*target_size, 6)
    label_shape = target_size

    # Define Petastorm schema for image and label data
    InputSchema = Unischema(
        "InputSchema",
        [
            UnischemaField("image", np.float32, input_shape, NdarrayCodec(), False),
            UnischemaField("label", np.uint8, label_shape, NdarrayCodec(), False),
        ],
    )

    try:
        # Step 4: Process each data split (train/validation/test)
        for split_name, split_df in datasets.items():
            if split_df.empty:
                continue

            profiler.log(f"Processing {split_name} split ({len(split_df)} patches)...")
    
            num_partitions, samples_per_partition = calculate_partitions(
                len(split_df), target_file_mb, target_size
            )
            
            profiler.log(f"Partitions: {num_partitions} (~{samples_per_partition} samples/partition, ~{target_file_mb}MB/file)")
            profiler.record(f"{split_name}_num_partitions", num_partitions)
            profiler.record(f"{split_name}_samples_per_partition", samples_per_partition)

            split_path = os.path.join(output_dir, split_name)
            if not split_path.startswith(("s3://", "s3a://")):
                os.makedirs(split_path, exist_ok=True)

            # Write Petastorm dataset using Spark
            with profiler.step(f"write_{split_name}_parquet"):
                with materialize_dataset(spark, split_path, InputSchema, target_file_mb):
                    rows_data = split_df.to_dict("records")
                    
                    rows_rdd = sc.parallelize(rows_data, num_partitions).mapPartitions(
                        lambda partition: process_partition(partition, InputSchema)
                    )
                    
                    rows_df = spark.createDataFrame(rows_rdd, InputSchema.as_spark_schema())
                    rows_df.write.mode("overwrite").parquet(split_path)

            output_paths[split_name] = split_path
            profiler.log(f"{split_name} dataset saved:  {split_path}")

    finally:
        # Step 5: Clean up Spark resources
        with profiler.step("spark_stop"):
            spark.stop()
            profiler.log("Spark session stopped.")

    # Step 6: Save profiling data
    profiler.save(output_dir, name=p_name)
    return output_paths


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert BigEarthNet patches to Petastorm format"
    )
    parser.add_argument(
        "--meta",
        type=str,
        default="s3a://ubs-homes/erasmus/raj/dlproject/metadata_with_paths.parquet",
        help="Metadata Parquet path",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="s3a://ubs-homes/erasmus/raj/dlproject/testpercent/petastorm",
        help="Output Petastorm dataset dir (S3 or local)",
    )
    parser.add_argument(
        "--p_name", type=str, default="conversion", help="Output profile name"
    )
    parser.add_argument(
        "--frac", type=float, default=0.001, help="Fraction of dataset to sample"
    )
    parser.add_argument(
        "--executor-mem", required=False, help="executor memory", default="4g"
    )
    parser.add_argument(
        "--driver-mem", required=False, help="driver memory", default="4g"
    )
    parser.add_argument("--core", type=int, default=2)
    parser.add_argument("--n_executor", type=int, default=2)
    parser.add_argument(
        "--target-file-mb", 
        type=int, 
        default=15, 
        help="Target parquet file size in MB"
    )
    args = parser.parse_args()

    start_time = time.time()
    convert_to_petastorm(
        metadata_path=args.meta,
        output_dir=args.out,
        fraction=args.frac,
        target_size=(120, 120),
        executor_mem=args.executor_mem,
        driver_mem=args.driver_mem,
        core=args.core,
        n_executor=args.n_executor,
        target_file_mb=args.target_file_mb,
        p_name=args.p_name,
        args_str=str(args),
    )
    end_time = time.time()
    print(f"Conversion completed in {end_time - start_time:.2f} seconds.")


if __name__ == "__main__":
    main()