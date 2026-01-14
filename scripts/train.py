"""Train semantic segmentation model on BigEarthNet using Petastorm."""

import argparse
import os
import tempfile

import boto3
import tensorflow as tf
from petastorm import make_reader
import warnings
warnings.filterwarnings('ignore', category=FutureWarning, module='petastorm') # petastorm warnings due to lib depreceation is causing me headache so lets supress this


def transform_row(row):
    """Transform Petastorm row into training format."""
    return row["input_data"], row["label"]


def build_unet_model():
    """Build U-Net style encoder-decoder for semantic segmentation."""
    return tf. keras.Sequential([
        tf. keras.layers.Input(shape=(120, 120, 6)),
        tf.keras.layers. Conv2D(64, 3, activation="relu", padding="same"),
        tf.keras.layers.Conv2D(64, 3, activation="relu", padding="same"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers. Conv2D(128, 3, activation="relu", padding="same"),
        tf.keras.layers.Conv2D(128, 3, activation="relu", padding="same"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers. Conv2D(256, 3, activation="relu", padding="same"),
        tf.keras.layers.UpSampling2D(),
        tf.keras.layers.Conv2D(128, 3, activation="relu", padding="same"),
        tf.keras.layers.UpSampling2D(),
        tf.keras.layers.Conv2D(64, 3, activation="relu", padding="same"),
        tf.keras.layers.Conv2D(256, 1, activation="softmax"),
    ])


def make_dataset(reader, batch_size, shuffle=True):
    """Create TF dataset from Petastorm reader."""
    dataset = tf. data.Dataset.from_generator(
        lambda: (dict(patch_id_int=x.patch_id_int, input_data=x.input_data, label=x.label) for x in reader),
        output_signature={
            "patch_id_int": tf.TensorSpec(shape=(), dtype=tf.int32),
            "input_data":  tf.TensorSpec(shape=(120, 120, 6), dtype=tf.float32),
            "label": tf.TensorSpec(shape=(120, 120), dtype=tf.uint8),
        },
    )
    dataset = dataset.map(transform_row, num_parallel_calls=tf.data.AUTOTUNE)
    if shuffle:
        dataset = dataset. shuffle(1000)
    dataset = dataset.batch(batch_size).repeat().prefetch(tf.data.AUTOTUNE)
    return dataset


def normalize_path(path):
    """Add file: // prefix if local path."""
    if not path.startswith(("s3://", "file://")):
        return f"file://{os.path.abspath(path)}"
    return path


def train_model(data_path, epochs=10, batch_size=32, lr=0.001):
    """Train segmentation model streaming from Petastorm dataset (local or S3)."""
    strategy = tf.distribute.MirroredStrategy()  # setup the gpu distribution strategy for multiple gpu per machine as shown in class

    train_path = normalize_path(os.path.join(data_path, "train"))
    val_path = normalize_path(os. path.join(data_path, "val"))
    test_path = normalize_path(os.path.join(data_path, "test"))

    print(f"Streaming data from: {data_path}")
    print(f"Number of gpu devices: {strategy. num_replicas_in_sync}")

    with make_reader(train_path, num_epochs=None, hdfs_driver="libhdfs3", reader_pool_type="thread") as train_reader, \
         make_reader(val_path, num_epochs=None, hdfs_driver="libhdfs3", reader_pool_type="thread") as val_reader, \
         make_reader(test_path, num_epochs=1, hdfs_driver="libhdfs3", reader_pool_type="thread") as test_reader:

        train_ds = strategy.experimental_distribute_dataset(make_dataset(train_reader, batch_size, shuffle=True))  # distributed dataset
        val_ds = strategy.experimental_distribute_dataset(make_dataset(val_reader, batch_size, shuffle=False))
        test_ds = strategy.experimental_distribute_dataset(make_dataset(test_reader, batch_size, shuffle=False))

        with strategy.scope(): # put the model creation and compilation inside the strategy scope so it would be distributed
            model = build_unet_model()
            model.compile(
                optimizer=tf.keras.optimizers. Adam(lr),
                loss="sparse_categorical_crossentropy",
                metrics=["accuracy"],
            )

        print(model. summary())

        callbacks = [
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        ]

        history = model. fit(
            train_ds,
            epochs=epochs,
            steps_per_epoch=38,
            validation_data=val_ds,
            validation_steps=10,
            callbacks=callbacks,
        )

        test_results = model.evaluate(test_ds, steps=10)
        print(f"\nTest loss: {test_results[0]:.4f}, Test accuracy: {test_results[1]:.4f}")

    print(f"Training complete! Final accuracy: {history.history['accuracy'][-1]:.4f}")
    return model, history


def main():
    parser = argparse.ArgumentParser(description="Train BigEarthNet semantic segmentation model")
    parser.add_argument("--data", required=True, help="Path to Petastorm dataset directory (contains train/ and test/ folders)")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--save", help="Path to save trained model (.keras)")

    args = parser.parse_args()
    model, _ = train_model(args.data, args. epochs, args.batch, args. lr)

    if args.save:
        if args.save. startswith("s3://"):
            with tempfile.TemporaryDirectory() as tmpdir:
                local_path = os.path.join(tmpdir, "model.keras")
                model.save(local_path)
                s3_path = args.save. replace("s3://", "")
                bucket, key = s3_path.split("/", 1)
                s3_client = boto3.client("s3")
                s3_client.upload_file(local_path, bucket, key)
                print(f"Model saved to {args.save}")
        else:
            model.save(args.save)
            print(f"Model saved to {args.save}")


if __name__ == "__main__": 
    main()