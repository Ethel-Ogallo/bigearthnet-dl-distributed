"""Train semantic segmentation model on BigEarthNet using Petastorm."""

import argparse
import json
import math
import multiprocessing
import os
import warnings

import s3fs
import tensorflow as tf
from petastorm import make_reader

from scripts.profiler import Profiler

tf.config.optimizer.set_jit(False)

os.environ["PYTHONWARNINGS"] = (
    "ignore::FutureWarning"  # we are using older version of petastorm & pyarrow , for god knows why , its in cluster so basically no option ! so i am supressing the warning for using depreceated APIs
)
warnings.filterwarnings("ignore", category=FutureWarning, module="petastorm")
warnings.filterwarnings("ignore", category=FutureWarning, module="pyarrow")


def record_metrics(profiler, **kwargs):
    for k, v in kwargs.items():
        profiler.record(k, v)


def log_gpu_info(profiler):
    gpus = tf.config.list_physical_devices("GPU")
    profiler.log(f"GPUs detected: {len(gpus)}")
    profiler.record("gpu_count", len(gpus))
    if gpus:
        for i, gpu in enumerate(gpus):
            try:
                profiler.log(
                    f"GPU {i}: {tf.config.experimental.get_device_details(gpu)}"
                )
            except Exception:
                pass
    else:
        profiler.log("WARNING: No GPUs - training on CPU")
    return len(gpus)


def build_model():
    inputs = tf.keras.layers.Input(shape=(120, 120, 6))
    x = tf.keras.layers.Conv2D(64, 3, activation="relu", padding="same")(inputs)
    x = tf.keras.layers.Conv2D(64, 3, activation="relu", padding="same")(x)
    x = tf.keras.layers.Add()(
        [x, tf.keras.layers.Conv2D(64, 1, padding="same")(inputs)]
    )
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Conv2D(128, 3, activation="relu", padding="same")(x)
    x = tf.keras.layers.Conv2D(128, 3, activation="relu", padding="same")(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Conv2D(256, 3, activation="relu", padding="same")(x)
    x = tf.keras.layers.UpSampling2D()(x)
    x = tf.keras.layers.Conv2D(128, 3, activation="relu", padding="same")(x)
    x = tf.keras.layers.UpSampling2D()(x)
    x = tf.keras.layers.Conv2D(64, 3, activation="relu", padding="same")(x)
    outputs = tf.keras.layers.Conv2D(45, 1, activation="softmax")(x)
    return tf.keras.Model(inputs, outputs)


def make_dataset(path, batch_size, dataset_size, shuffle=True):
    def gen():
        reader_kwargs = {
            "dataset_url": path,
            "num_epochs": None,
            "reader_pool_type": "thread",
            "workers_count": min(multiprocessing.cpu_count(), 8),
        }
        with make_reader(**reader_kwargs) as reader:
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
        dataset = dataset.shuffle(
            min(2000, dataset_size)
        )  # this is important for ram usage because shuffle gonna fill up buffer size n images from s3 to ram and send it to gpu with randomness

    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = (
        tf.data.experimental.AutoShardPolicy.DATA
    )  # tell tf to shard data across gpus no matter what the source is , shard from beginning
    dataset = dataset.with_options(options)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(
        tf.data.AUTOTUNE
    )  # let cpu prepare next batch while gpu is training on current batch , let tf figure it out by itself
    return dataset


def normalize_path(path):
    return (
        f"file://{os.path.abspath(path)}"
        if not path.startswith(("s3://", "file://"))
        else path
    )


def get_dataset_size(path, profiler):
    try:
        profiler.log(f"Reading dataset profile from: {path}")
        is_s3 = path.startswith(("s3://", "s3a://"))
        opener = s3fs.S3FileSystem(anon=False).open if is_s3 else open
        p = path.replace("s3a://", "s3://") if is_s3 else path.replace("file://", "")
        with opener(p, "r") as f:
            s = json.load(f)["summary"]
            return s["train_samples"], s["validation_samples"], s["test_samples"]
    except Exception as e:
        profiler.log(f"Metadata error: {e}")
        return None, None, None


def verify_s3_paths(base_path):
    if base_path.startswith(("s3://", "s3a://")):
        s3 = s3fs.S3FileSystem()
        base = base_path.replace("s3://", "").replace("s3a://", "").rstrip("/")
        for split in ["train", "validation", "test"]:
            if not s3.exists(f"{base}/{split}"):
                raise FileNotFoundError(f"Missing S3 path: {base}/{split}")


def train_model(
    data_path,
    epochs=10,
    batch_size=32,
    lr=0.001,
    p_name="train",
    args_str="",
    no_of_gpus=None,
    enable_lr_scaling=False,
):
    profiler = Profiler()
    profiler.log(f"Args: {args_str}")
    profiler.record("no_gpus_input", no_of_gpus)

    try:
        with profiler.step("gpu_setup"):
            log_gpu_info(profiler)

        with profiler.step("verify_paths"):
            verify_s3_paths(data_path)

        with profiler.step("strategy_init"):
            if no_of_gpus is not None:
                if enable_lr_scaling:
                    lr = lr * no_of_gpus
                    profiler.log(f"Adjusted learning rate: {lr}")
                profiler.record("learning_rate", lr)
                strategy = tf.distribute.MirroredStrategy(
                    devices=[f"GPU:{i}" for i in range(no_of_gpus)]
                )
            else:
                strategy = (
                    tf.distribute.MirroredStrategy()
                )  # works for multiple gpus on single machine
            profiler.log(f"Number of gpu devices: {strategy.num_replicas_in_sync}")
            global_batch = batch_size * strategy.num_replicas_in_sync
            profiler.log(
                f"Global batch: {global_batch} ({strategy.num_replicas_in_sync} replicas)"
            )
            record_metrics(
                profiler,
                strategy=strategy.__class__.__name__,
                num_replicas_in_sync=strategy.num_replicas_in_sync,
                batch_size_per_replica=batch_size,
                global_batch_size=global_batch,
            )

        with profiler.step("dataset_metadata"):
            train_path = normalize_path(os.path.join(data_path, "train"))
            val_path = normalize_path(os.path.join(data_path, "validation"))
            test_path = normalize_path(os.path.join(data_path, "test"))
            profile_path = normalize_path(
                os.path.join(data_path, "profile", "conversion_profile.json")
            )

            t_samples, v_samples, te_samples = get_dataset_size(profile_path, profiler)
            steps_per_epoch = math.ceil(t_samples / global_batch) if t_samples else 30
            val_steps = math.ceil(v_samples / global_batch) if v_samples else 10
            test_steps = math.ceil(te_samples / global_batch) if te_samples else 10
            samples_per_epoch = steps_per_epoch * global_batch
            profiler.log(f"samples/epoch: {samples_per_epoch}")
            profiler.log(f"steps/epoch: {steps_per_epoch}")
            record_metrics(
                profiler,
                train_samples=t_samples,
                validation_samples=v_samples,
                test_samples=te_samples,
                steps_per_epoch=steps_per_epoch,
                samples_per_epoch=samples_per_epoch,
            )

            with profiler.step("load_datasets"):
                with profiler.step("load_train"):
                    train_ds = strategy.experimental_distribute_dataset(
                        make_dataset(train_path, global_batch, t_samples, True)
                    )
                with profiler.step("load_validation"):
                    val_ds = strategy.experimental_distribute_dataset(
                        make_dataset(val_path, global_batch, v_samples, False)
                    )
                with profiler.step("load_test"):
                    test_ds = strategy.experimental_distribute_dataset(
                        make_dataset(test_path, global_batch, te_samples, False)
                    )

        with profiler.step("build_model"):
            with strategy.scope():
                model = build_model()
                model.compile(
                    optimizer=tf.keras.optimizers.Adam(lr),
                    loss="sparse_categorical_crossentropy",
                    metrics=["accuracy"],
                )
                model.summary(print_fn=profiler.log)

        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=5, restore_best_weights=True
            )
        ]

        with profiler.step("training", epochs=epochs):
            history = model.fit(
                train_ds,
                epochs=epochs,
                steps_per_epoch=steps_per_epoch,
                validation_data=val_ds,
                validation_steps=val_steps,
                callbacks=callbacks,
            )

        with profiler.step("evaluation"):
            test_loss, test_acc = model.evaluate(test_ds, steps=test_steps)
            profiler.log(f"Test Loss: {test_loss:.4f}, Acc: {test_acc:.4f}")
            record_metrics(profiler, test_loss=test_loss, test_accuracy=test_acc)

        profiler.save(data_path, name=p_name)
        return model, history
    finally:
        pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data", default="s3://ubs-homes/erasmus/raj/dlproject/testpercent/petastorm"
    )
    parser.add_argument("--p_name", default="train")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--gpus", type=int, default=None)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--enable_lr_scaling", default=False, action="store_true")
    args = parser.parse_args()
    train_model(
        args.data,
        args.epochs,
        args.batch,
        args.lr,
        args.p_name,
        str(args),
        args.gpus,
        args.enable_lr_scaling,
    )


if __name__ == "__main__":
    main()
