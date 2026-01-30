"""Convert BigEarthNet TIF files to Petastorm format for distributed deep learning.

This script orchestrates the complete data pipeline:
1. Reads metadata with S3 paths for Sentinel-1, Sentinel-2, and reference maps
2. Splits data into train/validation/test sets (maintaining existing splits)
3. Uses Apache Spark to parallelize TIF reading and processing from S3
4. Writes output in Petastorm format for efficient GPU training

Key design decisions:
- Uses bulk S3 reads (fs.cat) to fetch multiple files in parallel per partition
- Normalizes label classes using LABEL_MAPPING for consistent model training
- Configures Spark partitioning based on data size and executor resources
- Generates detailed performance profiles for conversion analysis
"""

import argparse
import json
import os
import platform
import time
import warnings
from contextlib import contextmanager
from datetime import datetime

import numpy as np
import psutil
import pyarrow.parquet as pq
import s3fs
from petastorm.codecs import NdarrayCodec
from petastorm.etl.dataset_metadata import materialize_dataset
from petastorm.unischema import Unischema, UnischemaField, dict_to_spark_row
from pyspark.sql import SparkSession
from rasterio.io import MemoryFile


def get_host_info():
    """Collect system information for profiling.

    Captures hardware configuration to understand performance characteristics
    and resource constraints during conversion.
    """
    info = {
        "hostname": platform.node(),
        "platform": platform.system(),
        "cpu_count": psutil.cpu_count(logical=True),
        "ram_gb": round(psutil.virtual_memory().total / (1024**3), 2),
    }
    try:
        import tensorflow as tf  # TensorFlow might not be available in Spark workers, so check conditionally

        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            info["cuda"] = {
                "device_count": len(gpus),
                "devices": [{"name": g.name} for g in gpus],
            }
    except ImportError:
        pass
    return info


def get_usage():
    """Capture current resource utilization for profiling.

    Monitors CPU, RAM, and GPU usage to identify bottlenecks during conversion.
    """
    usage = {
        "cpu_percent": psutil.cpu_percent(),
        "ram_percent": psutil.virtual_memory().percent,
        "ram_used_gb": round(psutil.virtual_memory().used / (1024**3), 2),
    }
    try:
        import tensorflow as tf

        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            gpu_usage = []
            for i, g in enumerate(gpus):
                try:
                    mem_info = tf.config.experimental.get_memory_info(g.name)
                    gpu_usage.append(
                        {
                            "device": i,
                            "memory_used_gb": round(mem_info["current"] / (1024**3), 2),
                            "memory_peak_gb": round(mem_info["peak"] / (1024**3), 2),
                        }
                    )
                except Exception:
                    gpu_usage.append({"device": i, "memory_used_gb": None})
            usage["cuda"] = gpu_usage
    except ImportError:
        pass
    return usage


class Profiler:
    def __init__(self):
        self.metrics = {
            "start_time": datetime.now().isoformat(),
            "host_info": get_host_info(),
            "steps": [],
            "summary": {},
        }
        self.step_stack = []
        self.log_messages = []

    @contextmanager
    def step(self, name, **meta):
        start = time.time()
        step_data = {"name": name, "start": start, **meta}
        self.step_stack.append(step_data)
        try:
            yield
        finally:
            duration = time.time() - start
            step_data["duration"] = duration
            step_data["end"] = time.time()
            self.step_stack.pop()
            self.metrics["steps"].append(
                {
                    "name": name,
                    "duration": duration,
                    "timestamp": datetime.fromtimestamp(start).isoformat(),
                    **meta,
                }
            )

    def log(self, message):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        usage = get_usage()
        usage_str = f"CPU:{usage['cpu_percent']}% RAM:{usage['ram_used_gb']}GB({usage['ram_percent']}%)"
        if "cuda" in usage:
            for g in usage["cuda"]:
                if g["memory_used_gb"] is not None:
                    usage_str += f" GPU{g['device']}:{g['memory_used_gb']}GB"
        log_entry = f"[{timestamp}] [{usage_str}] {message}"
        print(log_entry)
        self.log_messages.append(
            {"timestamp": timestamp, "message": message, "usage": usage}
        )

    def record(self, key, value):
        print(f"Recorded : {key} = {value}")
        self.metrics["summary"][key] = value

    def save(self, output_dir, name="profile"):
        import s3fs

        self.metrics["end_time"] = datetime.now().isoformat()
        total = sum(s["duration"] for s in self.metrics["steps"])
        self.metrics["summary"]["total_duration"] = total
        self.metrics["logs"] = self.log_messages

        is_s3 = output_dir.startswith(("s3://", "s3a://"))
        base_dir = output_dir.replace("s3a://", "s3://") if is_s3 else output_dir
        profile_dir = f"{base_dir}/profile"

        json_path = f"{profile_dir}/{name}_profile.json"
        log_path = f"{profile_dir}/{name}_profile.log"

        json_content = json.dumps(self.metrics, indent=2)
        log_lines = [f"Profile Report - {self.metrics['start_time']}\n{'='*60}\n"]

        hi = self.metrics["host_info"]
        log_lines.append(
            f"Host: {hi['hostname']} | CPU: {hi['cpu_count']} cores | RAM: {hi['ram_gb']}GB"
        )
        if "cuda" in hi:
            log_lines.append(f" | CUDA: {hi['cuda']['device_count']} GPU(s)")
        log_lines.append(f"\n{'='*60}\n")

        if self.log_messages:
            log_lines.append("\nLog Messages:\n")
            for entry in self.log_messages:
                u = entry["usage"]
                usage_str = f"CPU:{u['cpu_percent']}% RAM:{u['ram_used_gb']}GB"
                if "cuda" in u:
                    for g in u["cuda"]:
                        if g["memory_used_gb"] is not None:
                            usage_str += f" GPU{g['device']}:{g['memory_used_gb']}GB"
                log_lines.append(
                    f"[{entry['timestamp']}] [{usage_str}] {entry['message']}\n"
                )
            log_lines.append(f"\n{'='*60}\n")

        log_lines.append("\nStep Durations:\n")
        for step in self.metrics["steps"]:
            meta_str = ", ".join(
                f"{k}={v}"
                for k, v in step.items()
                if k not in ["name", "duration", "timestamp"]
            )
            meta_info = f" ({meta_str})" if meta_str else ""
            log_lines.append(f"{step['name']}{meta_info}:  {step['duration']:.2f}s\n")

        log_lines.append(f"\n{'='*60}\n")
        log_lines.append(
            f"Total:  {self.metrics['summary'].get('total_duration', 0):.2f}s\n"
        )

        for key, val in self.metrics["summary"].items():
            if key != "total_duration":
                log_lines.append(f"{key}: {val}\n")

        log_content = "".join(log_lines)

        if is_s3:
            fs = s3fs.S3FileSystem()
            with fs.open(json_path, "w") as f:
                f.write(json_content)
            with fs.open(log_path, "w") as f:
                f.write(log_content)
        else:
            import os

            os.makedirs(profile_dir, exist_ok=True)
            with open(json_path, "w") as f:
                f.write(json_content)
            with open(log_path, "w") as f:
                f.write(log_content)

        print(f"\nProfile saved:  {json_path}, {log_path}")


warnings.filterwarnings("ignore")

# BigEarthNet uses specific CORINE Land Cover class IDs (not sequential 0-44)
# We map them to sequential indices for efficient one-hot encoding in the model
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
# Create lookup table: CORINE class ID -> model class index (0-44)
LABEL_MAPPING = np.zeros(1000, dtype=np.uint8)
for idx, class_id in enumerate(BIGEARTH_IDS):
    LABEL_MAPPING[class_id] = idx


def process_partition(rows, schema):
    """Process partition with optimized bulk S3 reads.

    Why bulk reads: Fetching files one-by-one from S3 is slow due to latency.
    The s3fs.cat() API fetches multiple files in parallel, dramatically reducing
    total I/O time when processing thousands of satellite image patches.
    """
    # region_name specified because default region may fail for some AWS accounts
    fs = s3fs.S3FileSystem(anon=False, client_kwargs={"region_name": "eu-west-1"})

    for row in rows:
        try:

            def clean(p):
                # s3fs library doesn't support s3a:// protocol, only s3://
                return p.replace("s3a://", "").replace("s3://", "")

            # 1. Map logical names to S3 paths
            s2_bands = ["B02", "B03", "B04", "B08"]
            path_map = {
                "s1_vv": clean(f"{row['s1_path']}/{row['s1_name']}_VV.tif"),
                "s1_vh": clean(f"{row['s1_path']}/{row['s1_name']}_VH.tif"),
                "label": clean(
                    f"{row['reference_path']}/{row['patch_id']}_reference_map.tif"
                ),
            }
            for b in s2_bands:
                path_map[f"s2_{b}"] = clean(
                    f"{row['s2_path']}/{row['patch_id']}_{b}.tif"
                )

            # 2. Bulk fetch all files in parallel
            # fs.cat() fetches multiple S3 objects concurrently, significantly faster
            # than sequential reads. Returns dict {path: bytes}
            raw_files = fs.cat(list(path_map.values()))

            # 3. load results bytes to numpy in memory
            results = {}
            for key, path in path_map.items():
                with MemoryFile(raw_files[path]) as mem:
                    with mem.open() as ds:
                        results[key] = ds.read(1)

            # 4. Stack
            s1 = np.stack([results["s1_vv"], results["s1_vh"]], axis=-1)
            s2 = np.stack([results[f"s2_{b}"] for b in s2_bands], axis=-1)

            image = np.concatenate([s1, s2], axis=-1).astype(np.float32)
            # 5. Map CORINE labels to sequential indices for efficient model training
            # This normalization reduces memory and enables direct indexing
            label = LABEL_MAPPING[results["label"]].astype(np.uint8)

            yield dict_to_spark_row(schema, {"image": image, "label": label})

        except Exception as e:
            print(f"Skipping {row.get('patch_id', 'unknown')}: {e}")


def convert_to_petastorm(metadata_path, output_dir, fraction=1.0, args=None):
    """Convert BigEarthNet TIF files to Petastorm format using Apache Spark.

    Why Petastorm: Provides efficient data loading for TensorFlow/PyTorch with:
    - Columnar Parquet storage for fast I/O
    - Schema validation and type safety
    - Native integration with distributed training

    Why Spark: Enables parallel processing of 500k+ satellite images by distributing
    work across multiple executors, significantly reducing conversion time.
    """
    profiler = Profiler()
    profiler.log(
        f"Starting conversion with fraction {fraction}, {args.n_executor} exec"
    )

    # Read Metadata
    with profiler.step("read_metadata"):
        fs = s3fs.S3FileSystem(anon=False, client_kwargs={"region_name": "eu-west-1"})
        table = pq.read_table(metadata_path.replace("s3a://", "s3://"), filesystem=fs)
        df = table.to_pandas()

    # Split
    splits = {}
    for split in ["train", "validation", "test"]:
        sub_df = df[df["split"] == split]
        if fraction < 1.0:
            # Random sampling within each split to maintain distribution
            sub_df = sub_df.sample(frac=fraction, random_state=42)
        splits[split] = sub_df.reset_index(drop=True)
        profiler.record(f"{split}_samples", len(sub_df))

    # Init Spark
    with profiler.step("spark_init"):

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
            .config("spark.executor.memory", args.executor_mem)
            .config("spark.driver.memory", args.driver_mem)
            .config("spark.executor.cores", args.core)
            .config("spark.driver.maxResultSize", "512m")
            .config("spark.executor.instances", str(args.n_executor))
            .config(
                "spark.serializer", "org.apache.spark.serializer.KryoSerializer"
            )  # Kryo is more memory-efficient than Java serializer for binary data
            .config(
                "spark.sql.shuffle.partitions",
                str((args.core * args.n_executor) * 4),
            )  # rule of thumb : 2-4 partitions per core
            .config(
                "spark.sql.files.maxPartitionBytes", "268435456"
            )  # 256MB # intiial partition size
            .config(
                "spark.sql.adaptive.enabled", "true"
            )  # let spark optimize the shuffle partitions
            .config(
                "spark.sql.adaptive.advisoryPartitionSizeInBytes", "134217728"
            )  # 128MB # source : https://spark.apache.org/docs/latest/sql-performance-tuning.html
            .getOrCreate()
        )

    sc = spark.sparkContext
    schema = Unischema(
        "InputSchema",
        [
            UnischemaField("image", np.float32, (120, 120, 6), NdarrayCodec(), False),
            UnischemaField("label", np.uint8, (120, 120), NdarrayCodec(), False),
        ],
    )

    try:

        total_rows = sum(len(s) for s in splits.values())
        bytes_per_row = (120 * 120 * 6 * 4) + (
            120 * 120
        )  # Schema: 6 image bands (120x120, float32) + label (120x120, uint8)
        dataset_size_gb = round((total_rows * bytes_per_row) / (1024**3), 2)
        profiler.record("total_rows", total_rows)
        profiler.record("dataset_size_gb", dataset_size_gb)

        for name, split_df in splits.items():
            if split_df.empty:
                continue

            # internal spark partitions: based on data size with target_file_mb as partition size
            total_data_bytes = len(split_df) * bytes_per_row
            profiler.record(f"{name}_total_data_size_gb", total_data_bytes / (1024**3))

            spark_partitions = min(
                len(split_df), args.core * args.n_executor * 4
            )  # Ensure at least 4 partitions per core for efficient parallelism

            profiler.log(
                f"Processing {name}: {len(split_df)} rows, {spark_partitions} spark partitions"
            )

            out_path = os.path.join(output_dir, name)

            with profiler.step(f"write_{name}"):
                with materialize_dataset(spark, out_path, schema, args.target_file_mb):
                    rdd = sc.parallelize(split_df.to_dict("records"), spark_partitions)
                    rdd = rdd.mapPartitions(lambda x: process_partition(x, schema))

                    profiler.record(f"{name}_rdd_partitions", rdd.getNumPartitions())

                    df_spark = spark.createDataFrame(rdd, schema.as_spark_schema())
                    df_spark.write.mode("overwrite").parquet(out_path)

            profiler.log(f"Saved {name} to {out_path}")
            profiler.record(f"{name}_samples", len(split_df))
            tasks_per_executor = max(1, spark_partitions // args.n_executor)
            profiler.record(f"{name}_tasks_per_executor", tasks_per_executor)

    finally:
        spark.stop()
        profiler.save(output_dir, name=args.p_name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--meta",
        default="s3a://ubs-homes/erasmus/raj/dlproject/metadata_with_paths.parquet",
    )
    parser.add_argument(
        "--out", default="s3a://ubs-homes/erasmus/raj/dlproject/testpercent/petastorm"
    )
    parser.add_argument("--p_name", default="conversion")
    parser.add_argument("--frac", type=float, default=0.001)
    parser.add_argument("--executor-mem", default="4g")
    parser.add_argument("--driver-mem", default="4g")
    parser.add_argument("--core", type=int, default=2)
    parser.add_argument("--n_executor", type=int, default=2)
    parser.add_argument("--target-file-mb", type=int, default=50)
    args = parser.parse_args()

    t0 = time.time()
    convert_to_petastorm(args.meta, args.out, args.frac, args)
    print(f"Total time: {time.time() - t0:.2f}s")


if __name__ == "__main__":
    main()
