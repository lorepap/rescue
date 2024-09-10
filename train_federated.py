# ----------------------------
# ONLY FOR JUPYTER NOTEBOOK:
# import only at beginning because otherwise
# it will slow down jupiter execution

#import nest_asyncio
#
#nest_asyncio.apply()
#print("Async mode ON")
# ----------------------------

import os
import sys
import math

import random
import numpy as np
import tensorflow as tf
import tensorflow_federated as tff
from tensorflow import keras
from data_manager.dataset_processing import quantile_clipping, exp_root_norm
from traffic_matrix_helper.helper import generate_training_data
from data_manager.federated_dataset_preprocessing import create_clients_datasets_list
from data_manager.dataset_processing import concatenate_chunks, quantile_clipping, exp_root_norm
from trainers.train_federated import EdsrFederatedTrainer
from traffic_matrices_controller.traffic_matrices_manager import TrafficMatrixManager
from memory_manager.gpu_manager import set_gpus_used
from argparse import ArgumentParser
from sklearn.model_selection import train_test_split
from utils.misc import minmax_scale

from config import config_federated as config
import datetime

tf.get_logger().setLevel('INFO')

print(tf.__version__)
print(tff.__version__)

# GPU Settings
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    set_gpus_used(config.AWS_GPU_IDS)

# print("Executing tff on: ", [gpu for gpu in physical_devices])
# tff.backends.native.set_local_execution_context(client_tf_devices=physical_devices)

def main(args):

    # Initialization and Configuration
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    TRAFFIC_MATRICES_BASE_PATH = os.path.join(ROOT_DIR, config.DATA_FOLDER)
    
    # Input parameters
    SCALE_FACTOR = args.scale_factor
    WINDOW_SIZE = args.w_size
    AGGREGATION_FREQ = args.agg_freq
    NUM_MAX_CLIENTS = args.n_clients

    # Other parameters
    NUM_HR_PIXELS = config.TM_ORIGINAL_SIZE - config.TM_ORIGINAL_SIZE % SCALE_FACTOR
    path_to_train = os.path.join(config.DATA_FOLDER,f"data/train_data_{SCALE_FACTOR}x_{AGGREGATION_FREQ}_{WINDOW_SIZE}wsize")
    path_to_test = os.path.join(config.DATA_FOLDER,f"data/test_data_{SCALE_FACTOR}x_{AGGREGATION_FREQ}_{WINDOW_SIZE}wsize")
    path_to_val = os.path.join(config.DATA_FOLDER,f"data/val_data_{SCALE_FACTOR}x_{AGGREGATION_FREQ}_{WINDOW_SIZE}wsize")
    
    # Load hr traffic matrices
    hr_original_size_path = TRAFFIC_MATRICES_BASE_PATH + f"traffic_matrices_{AGGREGATION_FREQ}_pod"
    hr_original_size_path_shuffled = TRAFFIC_MATRICES_BASE_PATH + f"traffic_matrices_{AGGREGATION_FREQ}_pod_shuffled"
    hr_original = concatenate_chunks(hr_original_size_path_shuffled)
    hr_original = quantile_clipping(hr_original, config.QUANTILE_PERCENTAGE)
    hr_original = exp_root_norm(hr_original, config.NORMALIZATION_EXP)
    hr_original = minmax_scale(hr_original)

    # randomly select MAX_DATA_SIZE number of hr_original images
    hr_original = np.array(random.sample(list(hr_original), config.MAX_DATA_SIZE))

    # Split the data into train, test, and validation sets
    train_data, test_data = train_test_split(hr_original, test_size=0.2, random_state=42)
    train_data, val_data = train_test_split(train_data, test_size=0.2, random_state=42)

    # Generate low resolution images and augment training data (cropping and windowing)
    lr_images, hr_images = generate_training_data(train_data, crop_size=config.CROP_SIZE, scale_factor=SCALE_FACTOR, window_size=WINDOW_SIZE, use_edsr=True)
    lr_images_val, hr_images_val = generate_training_data(val_data, crop_size=config.CROP_SIZE, scale_factor=SCALE_FACTOR, window_size=WINDOW_SIZE, use_edsr=True)
    lr_images_test, hr_images_test = generate_training_data(test_data, crop_size=config.CROP_SIZE, scale_factor=SCALE_FACTOR, window_size=WINDOW_SIZE, use_edsr=True)

    # Convert to tf.data.Dataset
    train_data = tf.data.Dataset.from_tensor_slices((lr_images, hr_images))
    val_data = tf.data.Dataset.from_tensor_slices((lr_images_val, hr_images_val))
    test_data = tf.data.Dataset.from_tensor_slices((lr_images_test, hr_images_test))

    # Print shape of train, val and test datasets
    print("Train dataset shape: ", lr_images.shape, hr_images.shape)
    print("Validation dataset shape: ", lr_images_val.shape, hr_images_val.shape)
    print("Test dataset shape: ", lr_images_test.shape, hr_images_test.shape)

    # Split the training set among the clients
    # For each round, each client will have TOTAL_SIZE//N_clients number of samples
    print("Client training data")
    client_data_list = create_clients_datasets_list(args.n_clients, train_data)
    print("Client test data")
    client_test_list = create_clients_datasets_list(args.n_clients, test_data)

    # Number of batches per client will be the number of rounds
    num_batches = math.ceil((len(train_data) / args.n_clients) / config.BATCH_SIZE)

    # Path to save weights, models, logs, plots and checkpoints
    weights_dir = 'weights/edsr-federated'
    models_dir = 'models_json/edsr-federated'
    logs_dir = 'loss_logs/edsr-federated'
    plots_dir = 'model_plots/edsr-federated'
    checkpoints_dir = 'model_checkpoints/edsr-federated'

    SAVE_LOGS = True

    SAVE_BEST_ONLY = True
    INPUT_SPEC = client_data_list[0].element_spec

    # Name of the model for saving purposes
    model_descr_str = f"model-edsr-federated-windowsize{WINDOW_SIZE}-clients{NUM_MAX_CLIENTS}-filters{config.NUM_FILTERS}-res{config.NUM_RES_BLOCKS}-x{SCALE_FACTOR}-freq{AGGREGATION_FREQ}"

    # Create the trainer
    edsr_federated_trainer = EdsrFederatedTrainer(
            INPUT_SPEC,
            scale=SCALE_FACTOR,
            num_filters=config.NUM_FILTERS,
            num_res_blocks=config.NUM_RES_BLOCKS,
            model_descr_str=model_descr_str,
            weights_dir=weights_dir,
            models_dir=models_dir,
            logs_dir=logs_dir,
            plots_dir=plots_dir,
            checkpoints_dir=checkpoints_dir
    )

    # Error from: client_keras_optimizer_fn=lambda: tf.keras.optimizers.Adam(learning_rate=1e-3),
    # Error most likely due to the boundaries value that is too large
    # Getting back to a simple Adam optimizer with no schedule policy
    # TODO: learning rate should be tuned. First attempt 1e-3 
    # TODO: early stopping
    edsr_federated_trainer.initialize_federated_averaging_process(
                client_keras_optimizer_fn=lambda: tf.keras.optimizers.Adam(learning_rate=1e-3),
                server_keras_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=1.0),
            )

    # Train the model
    # Here we pass the centralized validation set
    edsr_federated_trainer.train(
                client_data_list,
                client_test_list,
                val_data,
                num_rounds= num_batches,
                batch_size=config.BATCH_SIZE,
                percentage_clients_epoch=config.PERCENTAGE_CLIENTS_EPOCH,
                earlystopping=config.EARLYSTOPPING,
                patience=config.FED_PATIENCE,
                min_delta=config.MIN_DELTA,
                save_logs=SAVE_LOGS
            )

    edsr_federated_trainer.keras_evaluate(
                test_data.batch(config.BATCH_SIZE),
    )

    #     edsr_federated_trainer.federated_evaluation(
    #         clients_test_datasets_list,
    #         percentage_clients_epoch=PERCENTAGE_CLIENTS_EPOCH
    #     )

    edsr_federated_trainer.save_model_json()
    edsr_federated_trainer.print_history()

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--scale_factor", "-s", type=int, default=2, help="Scale factor to resize image, default 2")
    parser.add_argument("--agg_freq", "-f", type=str, default="10S", help="TM timestamp aggregation frequency, default 10S")
    parser.add_argument("--w_size", "-w", type=int, default=108, help="Window size for image cropping and data augmentation, default 108")
    parser.add_argument("--n_clients", "-n", type=int, default=2, help="Number of clients for the federated learning")
    parser.add_argument("--load_data", action="store_true", help="Load saved datasets instead of creating them from scratch")
    parser.add_argument("--save_data", action="store_true", help="Save created datasets for later use")

    args = parser.parse_args()

    main(args)
    