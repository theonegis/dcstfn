import argparse
from pathlib import Path
import numpy as np
from osgeo import gdal_array
from math import sqrt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from skimage.measure import compare_psnr, compare_ssim


def evaluate(y_true, y_pred, func):
    assert y_true.shape == y_pred.shape
    if y_true.ndim == 2:
        y_true = y_true[np.newaxis, :]
        y_pred = y_pred[np.newaxis, :]
    metrics = []
    for i in range(y_true.shape[0]):
        metrics.append(func(y_true[i], y_pred[i]))
    return metrics


def mae(y_true, y_pred):
    return evaluate(y_true, y_pred,
                    lambda x, y: mean_absolute_error(x.ravel(), y.ravel()))


def rmse(y_true, y_pred):
    return evaluate(y_true, y_pred,
                    lambda x, y: sqrt(mean_squared_error(x.ravel(), y.ravel())))


def r2(y_true, y_pred):
    return evaluate(y_true, y_pred,
                    lambda x, y: r2_score(x.ravel(), y.ravel()))


def kge(y_true, y_pred):
    def compute(x, y):
        im_true = x.ravel()
        im_pred = y.ravel()
        r = np.corrcoef(im_true, im_pred)[1, 0]
        m_true = np.mean(im_true)
        m_pred = np.mean(im_pred)
        std_true = np.std(im_true)
        std_pred = np.std(im_pred)
        return 1 - np.sqrt((r - 1) ** 2
                           + (std_pred / std_true - 1) ** 2
                           + (m_pred / m_true - 1) ** 2)

    return evaluate(y_true, y_pred, compute)


def psnr(y_true, y_pred, data_range=10000):
    return evaluate(y_true, y_pred,
                    lambda x, y: compare_psnr(x, y, data_range=data_range))


def ssim(y_true, y_pred, data_range=10000):
    return evaluate(y_true, y_pred,
                    lambda x, y: compare_ssim(x, y, data_range=data_range))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # 输入数据为真实数据和预测数据
    parser.add_argument('inputs', nargs='+', type=Path)
    args = parser.parse_args()
    inputs = args.inputs

    assert len(inputs) == 2

    ix = gdal_array.LoadFile(str(inputs[0].expanduser().resolve()))
    iy = gdal_array.LoadFile(str(inputs[1].expanduser().resolve()))
    print('RMSE: ', *rmse(ix, iy))
    print('R2: ', *r2(ix, iy))
    print('KGE: ', *kge(ix, iy))
    print('SSIM: ', *ssim(ix, iy))
