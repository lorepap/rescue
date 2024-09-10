import numpy as np
import tensorflow as tf
import os

from models.edsr import edsr
from utils.misc import save_df_to_csv
from utils.misc import create_directory
from utils.plotting.tm_lr_hr_sr import plot_tm

# @TODO add windowsize as input to loading functions
def evaluate_metrics(mode, lr, hr, max_val_normalized, max_val_quantile, scale_factor, num_filters, num_res_block,
                     num_clients=None, percentage_clients=None, windowsize=None, submatricessize=None,
                     external_model=None, save_csv_path=None, plots_path=None, num_plots=0, plot_title=None, agg_freq=None):
    if mode == "bicubic":
        sr = bic(lr, scale_factor)
        model_name = mode + f"_x{scale_factor}"
    else:
        if external_model:
            model_instance = external_model
        else:
            if mode == "centralized":
                model_weights = load_weights_centralized(num_filters, num_res_block, scale_factor, windowsize, agg_freq)
                model_name = f"model-edsr-centralized-windowsize{windowsize}-filters{num_filters}-res{num_res_block}-x{scale_factor}_freq{agg_freq}"

            elif mode == "federated-performance":
                model_weights = load_weights_federated(num_clients, num_filters, num_res_block, scale_factor, windowsize, agg_freq)
                model_name = f"model-edsr-federated-windowsize{windowsize}-clients{num_clients}-filters{num_filters}-res{num_res_block}-x{scale_factor}_freq{agg_freq}"

            elif mode == "federated-privacy":
                model_weights = load_weights_federated_privacy(num_clients, windowsize, submatricessize, num_filters,
                                                               num_res_block, scale_factor)
                model_name = f"model-edsr-federated-privacy-clients{num_clients}-windowsize{windowsize}-submatricessize{submatricessize}-filters{num_filters}-res{num_res_block}-x{scale_factor}"

            elif mode == "federated-rnd":
                percentage_str = str("{:.2f}".format(percentage_clients * 100.)).replace('.', '_') # type: ignore
                model_weights = load_weights_federated_rnd(num_clients, percentage_str, windowsize, num_filters,
                                                           num_res_block, scale_factor)
                model_name = f"model-edsr-federated-performance-clients{num_clients}-percentage_clients{percentage_str}-windowsize{windowsize}-filters{num_filters}-res{num_res_block}-x{scale_factor}_freq{agg_freq}"

            else:
                raise ValueError("The value of 'mode' is not valid")

            model_instance = edsr(scale=scale_factor, input_depth=1, num_filters=num_filters,
                                  num_res_blocks=num_res_block,
                                  res_block_scaling=None)
            model_instance.load_weights(model_weights)

        sr = model_instance.predict(lr)

    lr_rec = ((lr * max_val_normalized) ** 2)  # Square-root normalization
    hr_rec = ((hr * max_val_normalized) ** 2)  # Square-root normalization
    sr_rec = ((sr * max_val_normalized) ** 2)  # Square-root normalization

    # Case of logarithmic normalization 
    #     hr_rec = (np.exp(hr*max_val_normalized)-1)
    #     sr_rec = (tf.math.exp(sr*max_val_normalized)-1)

    # Case of not normalization
    #     hr_rec = hr * max_val_normalized
    #     sr_rec = sr * max_val_normalized

    mae_original = mae(hr_rec, sr_rec, axis=(1, 2))
    mae_normalized = mae(hr, sr, axis=(1, 2))

    psnr_original = tf.image.psnr(
        hr_rec, sr_rec, max_val=max_val_quantile, name=None
    )

    psnr_normalized = tf.image.psnr(
        hr, sr, max_val=1.0, name=None
    )

    ssim_original = tf.image.ssim(
        tf.convert_to_tensor(hr_rec, dtype=tf.float32), tf.convert_to_tensor(sr_rec, dtype=tf.float32),
        max_val=max_val_quantile, filter_size=11, filter_sigma=1.5, k1=0.01, k2=0.03
    )

    ssim_normalized = tf.image.ssim(
        tf.convert_to_tensor(hr, dtype=tf.float32), tf.convert_to_tensor(sr, dtype=tf.float32), max_val=1.0,
        filter_size=11, filter_sigma=1.5, k1=0.01, k2=0.03
    )

    eval_metrics = {
        "mae_original": list(mae_original),
        "mae_normalized": list(mae_normalized),
        "psnr_original": list(np.array(psnr_original)),
        "psnr_normalized": list(np.array(psnr_normalized)),
        "ssim_original": list(np.array(ssim_original)),
        "ssim_normalized": list(np.array(ssim_normalized))
    }

    if save_csv_path:
        save_df_to_csv(eval_metrics, path=save_csv_path)

    """if plots_path:
        hr_path = os.path.join(plots_path, f"hr_x{scale_factor}")
        lr_path = os.path.join(plots_path, f"lr_x{scale_factor}")
        sr_path = os.path.join(plots_path, model_name)

        create_directory(hr_path)
        create_directory(lr_path)
        create_directory(sr_path)

        for i in range(num_plots):
            plot_tm(hr[i], max_val=1, colorbar=True, title=plot_title, cbarlabel="Traffic volume",
                    save_path=os.path.join(hr_path, "hr_{:04}.png".format(i)))
            plot_tm(lr[i], max_val=1, colorbar=True, title=plot_title, cbarlabel="Traffic volume",
                    save_path=os.path.join(lr_path, "lr_{:04}.png".format(i)))
            plot_tm(sr[i], max_val=1, colorbar=True, title=plot_title, cbarlabel="Traffic volume",
                    save_path=os.path.join(sr_path, "sr_{:04}.png".format(i)))"""

    return eval_metrics


def mae(hr, sr, axis=None):
    return np.mean(np.abs(sr - hr), axis=axis).reshape([-1])


def bic(lr, scale_factor, antialias=True):
    sr = tf.image.resize(lr, [lr.shape[-2] * scale_factor, lr.shape[-2] * scale_factor], method="bicubic",
                         antialias=antialias)
    return sr


# TODO: create a class "WeightsLoader" to manage the loading of the weights-files
def load_weights_centralized(num_filters, num_res_block, scale_factor, windowsize, agg_freq):
    weights_base_path = f"models_training_logs/weights/edsr/model-edsr-centralized-windowsize{windowsize}-filters{num_filters}-res{num_res_block}-x{scale_factor}_freq{agg_freq}"
    model_weights = os.path.join(weights_base_path,
                                 f"model-edsr-centralized-windowsize{windowsize}-filters{num_filters}-res{num_res_block}-x{scale_factor}_freq{agg_freq}-final.h5")
    return model_weights


def load_weights_federated(num_clients, num_filters, num_res_block, scale_factor, windowsize, agg_freq):
    weights_base_path = "models_training_logs/weights/edsr-federated"
    model_weights_dir = os.path.join(weights_base_path,
                                     f"model-edsr-federated-windowsize{windowsize}-clients{num_clients}-filters{num_filters}-res{num_res_block}-x{scale_factor}_freq{agg_freq}")
    model_weights = os.path.join(model_weights_dir,
                                 next(e for e in os.listdir(model_weights_dir) if e.startswith("model-edsr")))
    return model_weights


def load_weights_federated_rnd(num_clients, percentage_str, window_size, num_filters, num_res_block, scale_factor):
    weights_base_path = "./models_training_logs/weights/edsr-federated"
    model_weights_dir = os.path.join(os.path.realpath(weights_base_path),
                                     f"model-edsr-federated-performance-clients{num_clients}-percentage_clients{percentage_str}-windowsize{window_size}-filters{num_filters}-res{num_res_block}-x{scale_factor}")
    model_weights = os.path.join(model_weights_dir,
                                 next(e for e in os.listdir(model_weights_dir) if e.startswith("model-edsr")))
    return model_weights


def load_weights_federated_privacy(num_clients, windowsize, submatricessize, num_filters, num_res_block, scale_factor):
    weights_base_path = "./models_training_logs/weights/edsr-federated"
    model_weights_dir = os.path.join(os.path.realpath(weights_base_path),
                                     f"model-edsr-federated-privacy-clients{num_clients}-windowsize{windowsize}-submatricessize{submatricessize}-filters{num_filters}-res{num_res_block}-x{scale_factor}")
    model_weights = os.path.join(model_weights_dir,
                                 next(e for e in os.listdir(model_weights_dir) if e.startswith("model-edsr")))
    return model_weights
