import os

import numpy as np
import tensorflow as tf
from argparse import ArgumentParser

from data_manager.dataset_processing import concatenate_chunks, quantile_clipping, exp_root_norm
from memory_manager.gpu_manager import set_gpus_used
from traffic_matrices_controller.traffic_matrices_manager import TrafficMatrixManager
from trainers.train_centralized import Trainer, ModelWrapper
import matplotlib.pyplot as plt
from config import config_centralized as config
from sklearn.model_selection import train_test_split
from traffic_matrix_helper.helper import generate_training_data
from utils.metrics import PSNR
from utils.misc import minmax_scale


def main(args):

    use_edsr = True if args.model=="edsr" else False

    # GPU Settings
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        set_gpus_used(config.AWS_GPU_IDS)

    # Initialization and Configuration
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    TRAFFIC_MATRICES_BASE_PATH = os.path.join(ROOT_DIR, config.DATA_FOLDER)
    
    # Input parameters
    WINDOW_SIZE = args.w_size
    AGGREGATION_FREQ = args.agg_freq
    SCALE_FACTOR = args.scale_factor
    
    # Model description string for saving
    model_descr_str = f"model-{args.model}-centralized-windowsize{WINDOW_SIZE}-filters{config.NUM_FILTERS}-res{config.NUM_RES_BLOCKS}-x{SCALE_FACTOR}_freq{AGGREGATION_FREQ}"
    
    # Load and preprocess data
    hr_original_size_path = TRAFFIC_MATRICES_BASE_PATH + f"traffic_matrices_{AGGREGATION_FREQ}_pod"
    hr_original_size_path_shuffled = TRAFFIC_MATRICES_BASE_PATH + f"traffic_matrices_{AGGREGATION_FREQ}_pod_shuffled"
    hr_original = concatenate_chunks(hr_original_size_path_shuffled)
    hr_original = quantile_clipping(hr_original, config.QUANTILE_PERCENTAGE)
    hr_original = exp_root_norm(hr_original, config.NORMALIZATION_EXP)
    hr_original = minmax_scale(hr_original)

    # Split the data into train, test, and validation sets
    train_data, test_data = train_test_split(hr_original, test_size=0.2, random_state=42)
    train_data, val_data = train_test_split(train_data, test_size=0.2, random_state=42)

    # Generate low resolution images and augment training data (cropping and windowing)
    lr_images, hr_images = generate_training_data(train_data, crop_size=config.CROP_SIZE, scale_factor=SCALE_FACTOR, window_size=WINDOW_SIZE, use_edsr=use_edsr)
    lr_images_val, hr_images_val = generate_training_data(val_data, crop_size=config.CROP_SIZE, scale_factor=SCALE_FACTOR, window_size=WINDOW_SIZE, use_edsr=use_edsr)
    lr_images_test, hr_images_test = generate_training_data(test_data, crop_size=config.CROP_SIZE, scale_factor=SCALE_FACTOR, window_size=WINDOW_SIZE, use_edsr=use_edsr)

    # Print shape of train, val and test datasets
    print("Train dataset shape: ", lr_images.shape, hr_images.shape)
    print("Validation dataset shape: ", lr_images_val.shape, hr_images_val.shape)
    print("Test dataset shape: ", lr_images_test.shape, hr_images_test.shape)

    # Distribute over multiple GPUs (if available)
    strategy = tf.distribute.MirroredStrategy()
    
    # Create model
    with strategy.scope():
        wrapper = ModelWrapper(args.model, SCALE_FACTOR)
        model: tf.keras.Model = wrapper.model
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss='mean_absolute_error', metrics=PSNR)

    trainer = Trainer(wrapper, model_descr_str, save_logs=config.SAVE_LOGS)
    trainer.save_model_json()
    trainer.plot_model()
    print("[INFO] Training model", args.model)
    trainer.train((lr_images, hr_images), (lr_images_val, hr_images_val), epochs=config.EPOCHS, 
                       batch_size=config.BATCH_SIZE, min_delta=config.MIN_DELTA, patience=config.PATIENCE)

    # Evaluate model on test set
    print("[INFO] Evaluating model on test set")
    test_loss = model.evaluate(lr_images_test, hr_images_test)
    print(f"[INFO] Test loss: {test_loss}")
    
    # Save model
    trainer.save_model_weights()

    # Save history
    trainer.save_history()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--scale_factor", "-s", type=int, default=2, help="Scale factor to resize image, default 2")
    parser.add_argument("--agg_freq", "-f", type=str, default="10S", help="TM timestamp aggregation frequency, default 10S")
    parser.add_argument("--w_size", "-w", type=int, default=28, help="Window size for image cropping and data augmentation, default 28")
    parser.add_argument("--model", "-m", type=str, default="edsr", help="Model to use for training, default edsr")
    args = parser.parse_args()

    main(args)