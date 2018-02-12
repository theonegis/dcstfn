import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from osgeo import gdal_array
import numpy as np

parser = argparse.ArgumentParser()
# 输入数据为真实数据和预测数据
parser.add_argument('inputs', nargs='+', type=Path)
args = parser.parse_args()
inputs = args.inputs

assert len(inputs) == 2

scales = 10000
ix = gdal_array.LoadFile(str(inputs[0].expanduser().resolve()))
iy = gdal_array.LoadFile(str(inputs[1].expanduser().resolve()))
srx, sry = ix / scales, iy / scales

xx = srx.ravel()
yy = sry.ravel()

fit = np.polyfit(xx, yy, 1)
fit = np.poly1d(fit)


plt.figure()
plt.scatter(xx, yy, s=1, lw=0)
x = np.linspace(0, 1, num=1000)
plt.plot(x, fit(x))
plt.title("DCFN NIR Band")
plt.xlabel("Observed reflectance")
plt.ylabel("Predicted reflectance")
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.xticks(np.linspace(0, 1, num=10))
plt.yticks(np.linspace(0, 1, num=10))
plt.savefig('.'.join(['sr', 'png']), dpi=300)
plt.close()


