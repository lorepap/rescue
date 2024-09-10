import tensorflow as tf 

def PSNR(y_true, y_pred, max_val=1.0):
    return tf.image.psnr(y_true, y_pred, max_val=max_val)