import argparse
from pathlib import Path
import numpy as np
from scipy.stats import gaussian_kde
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from osgeo import gdal_array
import seaborn as sns
sns.set_context("paper", rc={'font.sans-serif': 'Arial',
                             'font.size': 12})

parser = argparse.ArgumentParser()
parser.add_argument('--true', '-t', type=Path, required=True,
                    help='the true observation data path')
parser.add_argument('--predict', '-p', type=Path, required=True,
                    help='the prediction data path')
parser.add_argument('--band', '-b', type=int, required=True,
                    help='the indicator for spectral band (0 for green, 1 for red, 2 for nir)')
parser.add_argument('--title', '-n', type=str, required=True,
                    help='the title of the image')
parser.add_argument('--output', '-o', type=str, required=True,
                    help='the output image file')
args = parser.parse_args()
true_file = args.true.expanduser()
pred_file = args.predict.expanduser()
band_ix = args.band
title = args.title
output_name = args.output

ix = gdal_array.LoadFile(str(true_file))
iy = gdal_array.LoadFile(str(pred_file))

if ix.ndim == 3:
    ix = ix[band_ix]
    iy = iy[band_ix]
# 单波段数据
assert ix.ndim == 2 and iy.ndim == 2

x = ix[:500, :500].flatten()
y = iy[:500, :500].flatten()
r2 = r2_score(x, y)

xy = np.vstack([x, y])
z = gaussian_kde(xy)(xy)
idx = z.argsort()
x, y, z = x[idx], y[idx], z[idx]

fig = plt.figure()
ax = plt.gca()
ax.scatter(x, y, c=z, s=1, cmap=plt.cm.rainbow)

max_sr = 3000 if band_ix in (0, 1) else 6000
ax.set_xlim((0, max_sr))
ax.set_ylim((0, max_sr))
ax.plot([0, max_sr], [0, max_sr], linewidth=1, color='gray')

ax.set_title(title, fontsize=14, fontweight='bold')
band_names = ['Green band', 'Red band', 'NIR band']
ax.text(max_sr * 0.1, max_sr * 0.9, band_names[band_ix], fontsize=10)
ax.text(max_sr * 0.8, max_sr * 0.1, r'$R^2=$' + '{:.3f}'.format(r2), fontsize=10)
ax.set_xlabel("Observed reflectance", fontsize=12)
ax.set_ylabel("Predicted reflectance", fontsize=12)
fig.savefig(output_name, dpi=900)
plt.close()
