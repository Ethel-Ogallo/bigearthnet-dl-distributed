"""Train semantic segmentation model on BigEarthNet using Petastorm."""

import argparse
import json
import os
import warnings

import tensorflow as tf
from petastorm import make_reader

from scripts.profiler import Profiler

warnings.filterwarnings("ignore", category=FutureWarning, module="petastorm")


def print_gpu_info():
    """Detect and print GPU information for training"""
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
    """Build U-Net semantic segmentation model"""
    return tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(120, 120, 6)),
            # Encoder
            tf.keras.layers.Conv2D(64, 3, activation="relu", padding="same"),
            tf.keras.layers.Conv2D(64, 3, activation="relu", padding="same"),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(128, 3, activation="relu", padding="same"),
            tf.keras.layers.Conv2D(128, 3, activation="relu", padding="same"),
            tf.keras.layers.MaxPooling2D(),
            # Bottleneck
            tf.keras.layers.Conv2D(256, 3, activation="relu", padding="same"),
            # Decoder
            tf.keras.layers.UpSampling2D(),
            tf.keras.layers.Conv2D(128, 3, activation="relu", padding="same"),
            tf.keras.layers.UpSampling2D(),
            tf.keras.layers.Conv2D(64, 3, activation="relu", padding="same"),
            # Output
            tf.keras.layers.Conv2D(
                256, 1, activation="softmax"
            ),  # there are only 45 classes but due to class naming we are keepin 256 here later on for efficient network it is worth mapping the classes before hand , here we would treat rest class as dummy class
        ]
    )


# def make_dataset(path, epochs, batch_size, shuffle=True):
#     """Create TensorFlow dataset from Petastorm data"""

#     def gen():
#         with make_reader(
#             path,
#             # num_epochs=epochs, # let petastorm reader supply the data continiously
#             hdfs_driver="libhdfs3",
#             reader_pool_type="thread",
#             workers_count=4,
#         ) as reader:
#             for sample in reader:
#                 yield sample.image, sample.label

#     dataset = tf.data.Dataset.from_generator(
#         gen,
#         output_signature=(
#             tf.TensorSpec(shape=(120, 120, 6), dtype=tf.float32),
#             tf.TensorSpec(
#                 shape=(120, 120), dtype=tf.uint8
#             ),  # we have the value until 999 on class name , so uint16 is required, but if we do 8 it should map the flooded value to 255
#         ),
#     )

#     if shuffle:
#         dataset = dataset.shuffle(2000)

#     dataset = dataset.batch(batch_size, drop_remainder=True)
#     dataset = (
#         dataset.repeat()
#     )  # let tensorflow control the epochs distribution of dataset
#     dataset = dataset.prefetch(tf.data.AUTOTUNE)

#     return dataset


def make_dataset(path, epochs, batch_size, shuffle=True):
    """Create TensorFlow dataset from Petastorm data"""

    def gen():
        with make_reader(
            path,
            num_epochs=epochs,
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
    """Convert local paths to file:// URIs"""
    if not path.startswith(("s3://", "file://")):
        return f"file://{os.path.abspath(path)}"
    return path


def get_dataset_size(path):
    """Read dataset sizes from conversion profile"""
    try:
        import s3fs

        s3_path = path.replace("s3a://", "s3://")
        fs = s3fs.S3FileSystem(anon=False)
        with fs.open(s3_path, "r") as f:
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
    """Verify that required S3 paths exist"""
    if base_path.startswith(("s3://", "s3a://")):
        import s3fs

        s3 = s3fs.S3FileSystem()
        clean_base = base_path.replace("s3://", "").replace("s3a://", "")

        for split in ["train", "validation", "test"]:
            path = f"{clean_base}/{split}"
            if not s3.exists(path):
                raise FileNotFoundError(f"S3 path does not exist: s3://{path}")
            print(f"Found s3://{path}")


def train_model(
    data_path, epochs=10, batch_size=32, lr=0.001, p_name="train", args_str=""
):
    """Train U-Net model on BigEarthNet Petastorm dataset"""
    profiler = Profiler()
    profiler.log(f"Args: {args_str}")

    # Step 1: Setup GPU configuration
    with profiler.step("gpu_setup"):
        num_gpus = print_gpu_info()

    profiler.record("num_gpus", num_gpus)

    # Step 2: Verify data paths exist
    with profiler.step("verify_paths"):
        if data_path.startswith("s3"):
            verify_s3_paths(data_path)

    # Step 3: Initialize distributed training strategy
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

    # Step 4: Load dataset metadata and compute steps
    with profiler.step("dataset_metadata"):
        train_path = normalize_path(os.path.join(data_path, "train"))
        val_path = normalize_path(os.path.join(data_path, "validation"))
        test_path = normalize_path(os.path.join(data_path, "test"))

        profile_path = normalize_path(
            os.path.join(data_path, "profile", "conversion_profile.json")
        )

        train_samples, val_samples, test_samples = get_dataset_size(profile_path)

        # Calculate steps per epoch based on dataset sizes
        steps_per_epoch = (
            (train_samples // global_batch_size) if train_samples else 38
        )  # https://datascience.stackexchange.com/questions/29719/how-to-set-batch-size-steps-per-epoch-and-validation-steps
        validation_steps = (val_samples // global_batch_size) if val_samples else 10
        test_steps = (test_samples // global_batch_size) if test_samples else 10

        print(
            f"Steps/epoch: {steps_per_epoch}, Validation:  {validation_steps}, Test: {test_steps}\n"
        )

    profiler.record("train_samples", train_samples)
    profiler.record("val_samples", val_samples)
    profiler.record("test_samples", test_samples)
    profiler.record("steps_per_epoch", steps_per_epoch)

    # Step 5: Load and prepare datasets
    with profiler.step("load_datasets"):
        train_ds = make_dataset(train_path, epochs, batch_size, shuffle=True)
        val_ds = make_dataset(val_path, epochs, batch_size, shuffle=False)
        test_ds = make_dataset(test_path, epochs, batch_size, shuffle=False)

        # Distribute datasets across devices
        train_ds = strategy.experimental_distribute_dataset(
            train_ds
        )  # this should handle the shards of dataset automatically
        val_ds = strategy.experimental_distribute_dataset(val_ds)
        test_ds = strategy.experimental_distribute_dataset(test_ds)

    # Step 6: Build and compile model
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

    # Step 7: Train the model
    with profiler.step("training", epochs=epochs):
        history = model.fit(
            train_ds,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            validation_data=val_ds,
            validation_steps=validation_steps,
            callbacks=callbacks,
        )

    # Step 8: Evaluate model on test set
    with profiler.step("evaluation"):
        test_loss, test_acc = model.evaluate(test_ds, steps=test_steps)
        print(f"\nTest Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")
        print(f"Final Train Accuracy: {history.history['accuracy'][-1]:.4f}\n")

    profiler.record("test_loss", float(test_loss))
    profiler.record("test_accuracy", float(test_acc))
    profiler.record("final_train_accuracy", float(history.history["accuracy"][-1]))
    profiler.record("epochs_completed", len(history.history["accuracy"]))

    # Step 9: Save profiling data
    profiler.save(data_path, name=p_name)

    return model, history


def main():
    parser = argparse.ArgumentParser(description="Train BigEarthNet segmentation model")
    parser.add_argument(
        "--data",
        default="s3://ubs-homes/erasmus/raj/dlproject/testpercent/petastorm",
        help="Petastorm dataset path (contains train/validation/test)",
    )
    parser.add_argument(
        "--p_name", type=str, default="train", help="Output profile name"
    )
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size per replica")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")

    args = parser.parse_args()
    train_model(
        args.data, args.epochs, args.batch, args.lr, args.p_name, args_str=str(args)
    )


if __name__ == "__main__":
    main()
