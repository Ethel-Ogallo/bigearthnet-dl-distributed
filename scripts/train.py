"""Train semantic segmentation model on BigEarthNet using Petastorm.
uv run scripts/train.py --data s3://ubs-homes/erasmus/ethel/bigearth/peta_trial_v2
"""

import argparse
import json
import os
import warnings

import tensorflow as tf
from petastorm import make_reader
from profiler import Profiler
from pyarrow import parquet as pq

warnings.filterwarnings("ignore", category=FutureWarning, module="petastorm")


def print_gpu_info():
    gpus = tf.config.list_physical_devices("GPU")
    print(f"\nGPUs detected: {len(gpus)}")

    if gpus:
        for i, gpu in enumerate(gpus):
            try:
                gpu_details = tf.config.experimental.get_device_details(gpu)
                print(f"GPU {i}: {gpu_details. get('device_name', 'Unknown')}")
            except:
                print(f"GPU {i}: info unavailable")
    else:
        print("WARNING: No GPUs - training on CPU")
    return len(gpus)


def build_unet_model():
    return tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(120, 120, 6)),
            tf.keras.layers.Conv2D(64, 3, activation="relu", padding="same"),
            tf.keras.layers.Conv2D(64, 3, activation="relu", padding="same"),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(128, 3, activation="relu", padding="same"),
            tf.keras.layers.Conv2D(128, 3, activation="relu", padding="same"),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(256, 3, activation="relu", padding="same"),
            tf.keras.layers.UpSampling2D(),
            tf.keras.layers.Conv2D(128, 3, activation="relu", padding="same"),
            tf.keras.layers.UpSampling2D(),
            tf.keras.layers.Conv2D(64, 3, activation="relu", padding="same"),
            tf.keras.layers.Conv2D(256, 1, activation="softmax"),
        ]
    )


def make_dataset(path, batch_size, shuffle=True):
    def gen():
        with make_reader(
            path,
            num_epochs=None,
            hdfs_driver="libhdfs3",
            reader_pool_type="thread",
            workers_count=4,
        ) as reader:
            for sample in reader:
                yield sample.image, sample.label

    dataset = tf.data.Dataset.from_generator(
        gen,
        output_signature=(
            tf.TensorSpec(shape=(120, 120, 6), dtype=tf.float32),
            tf.TensorSpec(shape=(120, 120), dtype=tf.uint8),
        ),
    )

    if shuffle:
        dataset = dataset.shuffle(2000)

    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


def normalize_path(path):
    if not path.startswith(("s3://", "file://")):
        return f"file://{os.path.abspath(path)}"
    return path


def get_dataset_size(path):
    try:
        import s3fs

        s3_path = path.replace("s3a://", "s3://")  # s3fs doesn't support s3a
        fs = s3fs.S3FileSystem(anon=False)
        with fs.open(s3_path, "r") as f:
            # print(f)
            data = json.load(f)
            return (
                data["summary"]["train_samples"],
                data["summary"]["validation_samples"],
                data["summary"]["test_samples"],
            )

    except Exception as e:
        print(f"Warning: Could not read metadata for {path}: {e}")
        return None, None, None


def verify_s3_paths(base_path):
    if base_path.startswith(("s3://", "s3a://")):
        import s3fs

        s3 = s3fs.S3FileSystem()
        clean_base = base_path.replace("s3://", "").replace("s3a://", "")

        for split in ["train", "validation", "test"]:
            path = f"{clean_base}/{split}"
            if not s3.exists(path):
                raise FileNotFoundError(f"S3 path does not exist: s3://{path}")
            print(f"Found s3://{path}")


def train_model(data_path, epochs=10, batch_size=32, lr=0.001):
    profiler = Profiler()

    with profiler.step("gpu_setup"):
        num_gpus = print_gpu_info()

    profiler.record("num_gpus", num_gpus)

    with profiler.step("verify_paths"):
        if data_path.startswith("s3"):
            verify_s3_paths(data_path)

    with profiler.step("strategy_init"):
        strategy = tf.distribute.MirroredStrategy()
        num_replicas = strategy.num_replicas_in_sync
        global_batch_size = batch_size * num_replicas

        print(f"Distribution strategy: {strategy.__class__.__name__}")
        print(f"Number of replicas: {num_replicas}")
        print(f"Batch size per replica: {batch_size}")
        print(f"Global batch size:  {global_batch_size}\n")

    profiler.record("num_replicas", num_replicas)
    profiler.record("batch_size_per_replica", batch_size)
    profiler.record("global_batch_size", global_batch_size)

    with profiler.step("dataset_metadata"):
        train_path = normalize_path(os.path.join(data_path, "train"))
        val_path = normalize_path(os.path.join(data_path, "validation"))
        test_path = normalize_path(os.path.join(data_path, "test"))

        profile_path = normalize_path(
            os.path.join(data_path, "profile", "conversion_profile.json")
        )

        train_samples, val_samples, test_samples = get_dataset_size(profile_path)

        # steps_per_epoch = (train_samples // global_batch_size) if train_samples else 38
        # validation_steps = (val_samples // global_batch_size) if val_samples else 10
        # test_steps = (test_samples // global_batch_size) if test_samples else 10

        steps_per_epoch = 30
        validation_steps = 10
        test_steps = 10

        print(
            f"Steps/epoch: {steps_per_epoch}, Validation:  {validation_steps}, Test: {test_steps}\n"
        )

    profiler.record("train_samples", train_samples)
    profiler.record("val_samples", val_samples)
    profiler.record("test_samples", test_samples)
    profiler.record("steps_per_epoch", steps_per_epoch)

    with profiler.step("load_datasets"):
        train_ds = make_dataset(train_path, batch_size, shuffle=True)
        val_ds = make_dataset(val_path, batch_size, shuffle=False)
        test_ds = make_dataset(test_path, batch_size, shuffle=False)

        train_ds = strategy.experimental_distribute_dataset(train_ds)
        val_ds = strategy.experimental_distribute_dataset(val_ds)
        test_ds = strategy.experimental_distribute_dataset(test_ds)

    with profiler.step("build_model"):
        with strategy.scope():
            model = build_unet_model()
            model.compile(
                optimizer=tf.keras.optimizers.Adam(lr),
                loss="sparse_categorical_crossentropy",
                metrics=["accuracy"],
            )
        print(model.summary())

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True
        ),
    ]

    with profiler.step("training", epochs=epochs):
        history = model.fit(
            train_ds,
            epochs=epochs,
            # steps_per_epoch=steps_per_epoch,
            validation_data=val_ds,
            # validation_steps=validation_steps,
            callbacks=callbacks,
        )

    with profiler.step("evaluation"):
        test_loss, test_acc = model.evaluate(test_ds)

        # test_loss, test_acc = model.evaluate(test_ds, steps=test_steps)
        print(f"\nTest Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")
        print(f"Final Train Accuracy: {history.history['accuracy'][-1]:.4f}\n")

    profiler.record("test_loss", float(test_loss))
    profiler.record("test_accuracy", float(test_acc))
    profiler.record("final_train_accuracy", float(history.history["accuracy"][-1]))
    profiler.record("epochs_completed", len(history.history["accuracy"]))

    profiler.save(data_path, name="train")

    return model, history


def main():
    parser = argparse.ArgumentParser(description="Train BigEarthNet segmentation model")
    parser.add_argument(
        "--data",
        default="s3://ubs-homes/erasmus/raj/dlproject/testpercent/petastorm",
        help="Petastorm dataset path (contains train/validation/test)",
    )
    parser.add_argument("--epochs", type=int, default=5, help="Training epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size per replica")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")

    args = parser.parse_args()
    train_model(args.data, args.epochs, args.batch, args.lr)


if __name__ == "__main__":
    main()
