from keras import backend as K
import tensorflow as tf
import numpy as np


def cov(x, y):
    return K.mean((x - K.mean(x)) * K.transpose((y - K.mean(y))))


def psnr(y_true, y_pred, data_range=10000):
    """Peak signal-to-noise ratio averaged over samples and channels."""
    mse = K.mean(K.square(y_true - y_pred), axis=(-3, -2))
    return K.mean(20 * K.log(data_range / K.sqrt(mse)) / np.log(10))


def ssim(y_true, y_pred, data_range=10000):
    """structural similarity measurement system."""
    K1 = 0.01
    K2 = 0.03

    mu_x = K.mean(y_pred)
    mu_y = K.mean(y_true)

    sig_x = K.std(y_pred)
    sig_y = K.std(y_true)
    sig_xy = cov(y_true, y_pred)

    L = data_range
    C1 = (K1 * L) ** 2
    C2 = (K2 * L) ** 2

    return ((2 * mu_x * mu_y + C1) * (2 * sig_xy * C2) /
            (mu_x ** 2 + mu_y ** 2 + C1) * (sig_x ** 2 + sig_y ** 2 + C2))


def r2(y_true, y_pred):
    # mean函数调用了tensor的属性，不能直接是一个ndarray
    tf_true = y_true
    if not isinstance(y_true, tf.Tensor):
        tf_true = tf.convert_to_tensor(y_true)
    res = K.sum(K.square(y_true - y_pred))
    tot = K.sum(K.square(y_true - K.mean(tf_true)))
    return 1 - res / (tot + K.epsilon())
