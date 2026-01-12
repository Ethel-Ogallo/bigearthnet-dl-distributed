"""Train semantic segmentation model on BigEarthNet using Petastorm."""

import argparse
import os

import tensorflow as tf
from petastorm import make_reader


def transform_row(row):
    """Transform Petastorm row into training format."""
    input_data = row["input_data"]
    label = row["label"]
    return input_data, label


def build_unet_model():
    """Build U-Net style encoder-decoder for semantic segmentation."""
    return tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(120, 120, 14)),
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


def train_model(data_path, epochs=10, batch_size=32, lr=0.001):
    """Train segmentation model streaming from Petastorm dataset (local or S3)."""
    strategy = (
        tf.distribute.MultiWorkerMirroredStrategy()
    )  # setup the gpu distribution strategy for multiple gpu as shown in class

    if not data_path.startswith(("s3://", "file://")):
        data_path = f"file://{os.path.abspath(data_path)}"

    print(f"Streaming data from: {data_path}")
    print(f"Number of gpu devices: {strategy.num_replicas_in_sync}")

    with make_reader(
        data_path,
        num_epochs=epochs,
        hdfs_driver="libhdfs3",
        reader_pool_type="thread",
    ) as reader:

        def dataset_fn():
            dataset = tf.data.Dataset.from_generator(
                lambda: reader,
                output_signature={
                    "input_data": tf.TensorSpec(shape=(120, 120, 14), dtype=tf.float32),
                    "label": tf.TensorSpec(shape=(120, 120), dtype=tf.uint8),
                },
            )
            return (
                dataset.map(transform_row, num_parallel_calls=tf.data.AUTOTUNE)
                .shuffle(1000)
                .batch(batch_size)
                .prefetch(tf.data.AUTOTUNE)
            )

        dataset = strategy.experimental_distribute_dataset(
            dataset_fn()
        )  # distributed dataset

        with strategy.scope():
            model = build_unet_model()
            model.compile(
                optimizer=tf.keras.optimizers.Adam(lr),
                loss="categorical_crossentropy",
                metrics=["accuracy"],
            )

        print(model.summary())

        history = model.fit(
            dataset,
            epochs=epochs,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
            ],
        )

    print(f"\nTraining complete! Final accuracy: {history.history['accuracy'][-1]:.4f}")
    return model, history


def main():
    parser = argparse.ArgumentParser(
        description="Train BigEarthNet semantic segmentation model"
    )
    parser.add_argument(
        "--data",
        required=True,
        help="Path to Petastorm dataset (local path or s3://bucket/path)",
    )
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of training epochs"
    )
    parser.add_argument("--batch", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--save", help="Path to save trained model (.keras)")

    args = parser.parse_args()
    model, _ = train_model(args.data, args.epochs, args.batch, args.lr)

    if args.save:
        if args.save.startswith("s3://"):
            # Save to S3
            with tempfile.TemporaryDirectory() as tmpdir:
                local_path = os.path.join(tmpdir, "model.keras")
                model.save(local_path)

                # Upload to S3
                s3_path = args.save.replace("s3://", "")
                bucket, key = s3_path.split("/", 1)
                s3_client = boto3.client("s3")
                s3_client.upload_file(local_path, bucket, key)
                print(f"Model saved to {args.save}")
        else:
            # Save locally
            model.save(args.save)
            print(f"Model saved to {args.save}")


if __name__ == "__main__":
    main()
