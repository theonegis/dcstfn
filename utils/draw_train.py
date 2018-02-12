import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('metrics', type=Path)
args = parser.parse_args()
csv_metrics = args.metrics


df = pd.read_csv(str(csv_metrics))

epoch = df['epoch']
for metric in ['Loss', 'PSNR']:
    train = df[metric.lower()]
    val = df['val_' + metric.lower()]
    plt.figure()
    plt.plot(epoch, train, label='train')
    plt.plot(epoch, val, label='val')
    plt.legend(loc='best')
    plt.xlabel('Epoch')
    plt.xticks(range(1, epoch.size + 1, 2))
    plt.ylabel(metric)
    plt.savefig('.'.join([metric.lower(), 'eps']))
    plt.close()
