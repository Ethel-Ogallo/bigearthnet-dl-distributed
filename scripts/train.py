"""Train semantic segmentation model on BigEarthNet TFRecord dataset."""

import argparse
import glob
import os
import tempfile
import boto3
import tensorflow as tf
import tensorflow_io as tfio

def parse_tfrecord(serialized_example):
    """Parse and decode TFRecord example into tensors."""
    feature_spec = {
        'patch_id': tf.io.FixedLenFeature([], tf.string),
        's1_data': tf.io.FixedLenFeature([], tf.string),  # Sentinel-1 (VV, VH)
        's2_data': tf.io.FixedLenFeature([], tf.string),  # Sentinel-2 (12 bands)
        'label': tf.io.FixedLenFeature([], tf.string),    # CLC land cover codes
    }
    example = tf.io.parse_single_example(serialized_example, feature_spec)
    
    # Decode binary strings to arrays and reshape to original dimensions
    s1 = tf.reshape(tf.io.decode_raw(example['s1_data'], tf.float32), [120, 120, 2])
    s2 = tf.reshape(tf.io.decode_raw(example['s2_data'], tf.float32), [120, 120, 12])
    label = tf.reshape(tf.io.decode_raw(example['label'], tf.uint8), [120, 120])
    
    # Concatenate S1 and S2 bands into single input tensor (120x120x14)
    return tf.concat([s1, s2], axis=-1), label

def build_unet_model():
    """Build U-Net style encoder-decoder for semantic segmentation."""
    return tf.keras.Sequential([
        tf.keras.layers.Input(shape=(120, 120, 14)),
        # Encoder
        tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same'),
        tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same'),
        tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D(),
        # Bottleneck
        tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same'),
        # Decoder
        tf.keras.layers.UpSampling2D(),
        tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same'),
        tf.keras.layers.UpSampling2D(),
        tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same'),
        # Output: 256 classes (CLC codes) per pixel
        tf.keras.layers.Conv2D(256, 1, activation='softmax'),
    ])

def train_model(data_path, epochs=10, batch_size=32, lr=0.001):
    """Train segmentation model on TFRecord dataset."""
    # Load TFRecord files from local path or S3
    if data_path.startswith('s3://'):
        # Parse S3 path
        s3_path = data_path.replace('s3://', '')
        bucket, prefix = s3_path.split('/', 1)
        
        # List all .tfrecord files in S3
        s3_client = boto3.client('s3')
        response = s3_client.list_objects_v2(Bucket=bucket, Prefix=prefix)
        
        if 'Contents' not in response:
            raise ValueError(f"No files found in {data_path}")
        
        tfrecord_files = [
            f"s3://{bucket}/{obj['Key']}" 
            for obj in response['Contents'] 
            if obj['Key'].endswith('.tfrecord')
        ]
    else:
        # Local file system
        tfrecord_files = glob.glob(f"{data_path}/*.tfrecord")
    
    if not tfrecord_files:
        raise ValueError(f"No TFRecord files found in {data_path}")
    print(f"Found {len(tfrecord_files)} TFRecord files in {data_path}")
    
    # Build efficient data pipeline
    dataset = (tf.data.TFRecordDataset(tfrecord_files)
        .map(parse_tfrecord, num_parallel_calls=tf.data.AUTOTUNE)
        .shuffle(1000)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
        .repeat(epochs))
    
    # Build and compile model
    model = build_unet_model()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    print(model.summary())
    
    # Train with early stopping and learning rate reduction
    history = model.fit(
        dataset,
        epochs=epochs,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(patience=2, factor=0.5)
        ]
    )
    
    print(f"\nTraining complete! Final accuracy: {history.history['accuracy'][-1]:.4f}")
    return model, history

def main():
    parser = argparse.ArgumentParser(description='Train BigEarthNet semantic segmentation model')
    parser.add_argument('--data', required=True, help='Path to TFRecord data directory')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--save', help='Path to save trained model (.keras)')
    
    args = parser.parse_args()
    model, _ = train_model(args.data, args.epochs, args.batch, args.lr)
    
    if args.save:
        if args.save.startswith('s3://'):
            # Save to S3
            with tempfile.TemporaryDirectory() as tmpdir:
                local_path = os.path.join(tmpdir, 'model.keras')
                model.save(local_path)
                
                # Upload to S3
                s3_path = args.save.replace('s3://', '')
                bucket, key = s3_path.split('/', 1)
                s3_client = boto3.client('s3')
                s3_client.upload_file(local_path, bucket, key)
                print(f"Model saved to {args.save}")
        else:
            # Save locally
            model.save(args.save)
            print(f"Model saved to {args.save}")

if __name__ == "__main__":
    main()
