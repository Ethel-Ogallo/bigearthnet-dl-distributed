import argparse
import json
import os
import time
import warnings
from contextlib import contextmanager
from datetime import datetime

import numpy as np
import pyarrow.parquet as pq
import s3fs
from petastorm.codecs import NdarrayCodec
from petastorm.etl.dataset_metadata import materialize_dataset
from petastorm.unischema import Unischema, UnischemaField, dict_to_spark_row
from pyspark.sql import SparkSession
from rasterio.io import MemoryFile

# from scripts.profiler import Profiler

# from profiler import Profiler # run this if you are using docker standalone scripts


class Profiler:
    def __init__(self):
        self.metrics = {
            "start_time": datetime.now().isoformat(),
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
        log_entry = f"[{timestamp}] {message}"
        print(log_entry)
        self.log_messages.append(log_entry)

    def record(self, key, value):
        print(f"Recorded : {key} = {value}")
        self.metrics["summary"][key] = value

    def save(self, output_dir, name="profile"):
        import s3fs

        self.metrics["end_time"] = datetime.now().isoformat()
        total = sum(s["duration"] for s in self.metrics["steps"])
        self.metrics["summary"]["total_duration"] = total

        is_s3 = output_dir.startswith(("s3://", "s3a://"))
        base_dir = output_dir.replace("s3a://", "s3://") if is_s3 else output_dir
        profile_dir = f"{base_dir}/profile"

        json_path = f"{profile_dir}/{name}_profile.json"
        log_path = f"{profile_dir}/{name}_profile.log"

        json_content = json.dumps(self.metrics, indent=2)
        log_lines = [f"Profile Report - {self.metrics['start_time']}\n{'='*60}\n"]

        if self.log_messages:
            log_lines.append("\nLog Messages:\n")
            for msg in self.log_messages:
                log_lines.append(f"{msg}\n")
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
            f"Total:  {self.metrics['summary']. get('total_duration', 0):.2f}s\n"
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


def process_partition(rows, schema):
    """Process partition with optimized bulk S3 reads."""
    # region_name ; as it was failing in my local  for some reason
    fs = s3fs.S3FileSystem(anon=False, client_kwargs={"region_name": "eu-west-1"})

    for row in rows:
        try:

            def clean(p):
                return p.replace("s3a://", "").replace(
                    "s3://", ""
                )  # s3fs doesn't support s3a

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
            # fs.cat returns a dict {path: bytes} which is
            raw_files = fs.cat(
                list(path_map.values())
            )  # fs has this cool api that can fetch multiple files in parallel https://s3fs.readthedocs.io/en/latest/api.html#s3fs.core.S3FileSystem.cat

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
            # 5. Map labels, well as we are reading labels anyway so why not optimize all the way and normalize the label classes
            label = LABEL_MAPPING[results["label"]].astype(np.uint8)

            yield dict_to_spark_row(schema, {"image": image, "label": label})

        except Exception as e:
            print(f"Skipping {row.get('patch_id', 'unknown')}: {e}")


def convert_to_petastorm(metadata_path, output_dir, fraction=1.0, args=None):
    profiler = Profiler()
    profiler.log(f"Starting conversion with fraction {fraction}")

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
            )  # memory efficient serializer source : https://www.javaspring.net/blog/java-lang-outofmemoryerror-java-heap-space-spark/
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
        for name, split_df in splits.items():
            if split_df.empty:
                continue

            # calculate partitions , we need to control the size of the output files , initially we did by controlling the parition numbers but if we can approx estimate filesize why not do it by file size ?
            bytes_per_row = (120 * 120 * 6 * 4) + (
                120
                * 120  # * 4 is back again because image is float 32 and label is uint8 , so 8*4 ,
            )  # it is coming from above schema , 6 image bands with 120/120 image size and labels is basically 120/120
            rows_per_file = int(
                (args.target_file_mb * 1024**2) / (bytes_per_row * 0.4)
            )  # 0.4 compression factor, i am putting this as approximate , idk honestly how much is the compression factor as it depends upon the type of the data being compressed
            partitions = max(1, len(split_df) // rows_per_file)

            profiler.log(
                f"Processing {name}: {len(split_df)} rows, {partitions} partitions"
            )

            out_path = os.path.join(output_dir, name)

            with profiler.step(f"write_{name}"):
                with materialize_dataset(spark, out_path, schema, args.target_file_mb):
                    rdd = sc.parallelize(split_df.to_dict("records"), partitions)
                    rdd = rdd.mapPartitions(lambda x: process_partition(x, schema))

                    df_spark = spark.createDataFrame(rdd, schema.as_spark_schema())
                    df_spark.write.mode("overwrite").parquet(out_path)

            profiler.log(f"Saved {name} to {out_path}")
            profiler.record(f"{name}_samples", len(split_df))
            profiler.record(f"{name}_partitions", partitions)

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
