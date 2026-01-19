import os
import time

import numpy as np
import pyarrow.parquet as pq
import s3fs
from petastorm.codecs import NdarrayCodec
from petastorm.etl.dataset_metadata import materialize_dataset
from petastorm.unischema import Unischema, UnischemaField, dict_to_spark_row
from pyspark.sql import SparkSession
from rasterio.io import MemoryFile

from scripts.profiler import Profiler


def read_s3_tif(s3_path):
    """Read TIF file from S3 and return as numpy array"""
    s3_path = s3_path.replace("s3a://", "s3://")
    fs = s3fs.S3FileSystem(anon=False)
    with fs.open(s3_path, "rb") as f:
        with MemoryFile(f.read()) as memfile:
            with memfile.open() as dataset:
                return dataset.read()


def process_patch_stream(row_dict):
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
        file_data = {}
        for key, path in s3_paths.items():
            file_data[key] = read_s3_tif(path)[0]

        # Combine S1 data (VV and VH polarizations)
        s1_data = np.stack([file_data["s1_vv"], file_data["s1_vh"]], axis=-1).astype(
            np.float32
        )
        # Combine S2 data (4 bands)
        s2_data = np.stack(
            [file_data[f"s2_{band}"] for band in s2_bands], axis=-1
        ).astype(np.float32)
        # Concatenate S1 and S2 to create final image (120x120x6)
        image = np.concatenate([s1_data, s2_data], axis=-1).astype(np.float32)
        label = file_data["label"].astype(np.uint8)

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


def convert_to_petastorm(
    metadata_path,
    output_dir,
    fraction=1.0,
    target_size=(120, 120),
    executor_mem="8g",
    driver_mem="4g",
    core=4,
    n_executor=3,
    p_name="conversion",
    args_str="",
):
    """Convert BigEarthNet TIF files to Petastorm format using Spark"""
    profiler = Profiler()
    profiler.log(f"Args: {args_str}")

    # Step 1: Read metadata parquet file
    with profiler.step("read_metadata"):
        print(f"Reading metadata from {metadata_path}")
        table = pq.read_table(metadata_path.replace("s3a://", "s3://"))
        df = table.to_pandas()
        print(f"Total patches: {len(df)}")
    profiler.record("fraction", fraction)
    profiler.record("total_patches", len(df))

    # Step 2: Split data into train/val/test and sample fraction
    with profiler.step("split_and_sample", fraction=fraction):
        train_df, val_df, test_df = split_and_sample(df, fraction)
        datasets = {"train": train_df, "validation": val_df, "test": test_df}

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

            print(f"\nProcessing {split_name} split ({len(split_df)} patches)...")

            with profiler.step(f"process_{split_name}", patches=len(split_df)):
                rowgroup_size_mb = 256

                def row_generator(index):
                    row = split_df.iloc[index]
                    patch = process_patch_stream(row)
                    if patch:
                        return dict_to_spark_row(
                            InputSchema,
                            {"image": patch["image"], "label": patch["label"]},
                        )
                    return None

                split_path = os.path.join(output_dir, split_name)
                if not split_path.startswith(("s3://", "s3a://")):
                    os.makedirs(split_path, exist_ok=True)

                # Write Petastorm dataset using Spark
                with profiler.step(f"write_{split_name}_parquet"):
                    with materialize_dataset(
                        spark, split_path, InputSchema, rowgroup_size_mb
                    ):
                        # Parallelize data processing across Spark cluster
                        rows_rdd = (
                            sc.parallelize(range(len(split_df)))
                            .map(row_generator)
                            .filter(lambda x: x is not None)
                        )
                        # repartition
                        num_partitions = min(max(50, len(split_df) // 100), 400)

                        rows_df = spark.createDataFrame(
                            rows_rdd, InputSchema.as_spark_schema()
                        )
                        rows_df = rows_df.repartition(num_partitions)
                        rows_df.write.mode("overwrite").parquet(split_path)

                output_paths[split_name] = split_path
                print(f"{split_name} dataset saved:  {split_path}")

    finally:
        # Step 5: Clean up Spark resources
        with profiler.step("spark_stop"):
            spark.stop()
            print("Spark session stopped.")

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
        p_name=args.p_name,
        args_str=str(args),
    )
    end_time = time.time()
    print(f"Conversion completed in {end_time - start_time:.2f} seconds.")


if __name__ == "__main__":
    main()
