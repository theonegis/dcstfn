from functools import partial
from pathlib import Path
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from keras import backend as K
from keras.callbacks import CSVLogger, ModelCheckpoint
from keras.utils.vis_utils import plot_model
from keras.preprocessing.image import img_to_array

from osgeo import gdal_array

from toolbox.data import data_dir, load_image_pairs, load_test_set
from toolbox.metrics import psnr, r2


class Experiment(object):
    def __init__(self, scale=16, load_set=None, build_model=None,
                 optimizer='adam', save_dir='.'):
        self.scale = scale
        self.load_set = partial(load_set, scale=scale)
        self.build_model = partial(build_model)
        self.optimizer = optimizer
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.config_file = self.save_dir / 'config.yaml'
        self.model_file = self.save_dir / 'model.hdf5'
        self.visual_file = self.save_dir / 'model.eps'

        self.train_dir = self.save_dir / 'train'
        self.train_dir.mkdir(exist_ok=True)
        self.history_file = self.train_dir / 'history.csv'
        self.weights_dir = self.train_dir / 'weights'
        self.weights_dir.mkdir(exist_ok=True)

        self.test_dir = self.save_dir / 'test'
        self.test_dir.mkdir(exist_ok=True)

    def weights_file(self, epoch=None):
        if epoch is None:
            return self.weights_dir / 'ep{epoch:04d}.hdf5'
        else:
            return self.weights_dir / 'ep{:04d}.hdf5'.format(epoch)

    @property
    def latest_epoch(self):
        try:
            return pd.read_csv(str(self.history_file))['epoch'].iloc[-1]
        except (FileNotFoundError, pd.io.common.EmptyDataError):
            pass
        return -1

    @staticmethod
    def _ensure_dimension(array, dim):
        while len(array.shape) < dim:
            array = array[np.newaxis, ...]
        return array

    @staticmethod
    def _ensure_channel(array, c):
        return array[..., c:c + 1]

    @staticmethod
    def validate(array):
        array = Experiment._ensure_dimension(array, 4)
        array = Experiment._ensure_channel(array, 0)
        return array

    def compile(self, model):
        """Compile model with default settings."""
        model.compile(optimizer=self.optimizer, loss='mse', metrics=[psnr, r2])
        return model

    def train(self, train_set, val_set, epochs=10, resume=True):
        # Load and process data
        x_train, y_train = self.load_set(train_set)
        x_val, y_val = self.load_set(val_set)
        assert len(x_train) == 3 and len(x_val) == 3
        for i in range(3):
            x_train[i], x_val[i] = [self.validate(x) for x in [x_train[i], x_val[i]]]
        y_train, y_val = [self.validate(y) for y in [y_train, y_val]]

        # Compile model
        model = self.compile(self.build_model(*x_train))
        model.summary()
        self.config_file.write_text(model.to_yaml())
        plot_model(model, to_file=str(self.visual_file), show_shapes=True)

        # Inherit weights
        if resume:
            latest_epoch = self.latest_epoch
            if latest_epoch > -1:
                weights_file = self.weights_file(epoch=latest_epoch)
                model.load_weights(str(weights_file))
            initial_epoch = latest_epoch + 1
        else:
            initial_epoch = 0

        # Set up callbacks
        callbacks = []
        callbacks += [ModelCheckpoint(str(self.model_file))]
        callbacks += [ModelCheckpoint(str(self.weights_file()),
                                      save_weights_only=True)]
        callbacks += [CSVLogger(str(self.history_file), append=resume)]

        # Train
        model.fit(x_train, y_train, batch_size=320, epochs=epochs, callbacks=callbacks,
                  validation_data=(x_val, y_val), initial_epoch=initial_epoch)

        # Plot metrics history
        prefix = str(self.history_file).rsplit('.', maxsplit=1)[0]
        df = pd.read_csv(str(self.history_file))
        epoch = df['epoch']
        for metric in ['Loss', 'PSNR', 'R2']:
            train = df[metric.lower()]
            val = df['val_' + metric.lower()]
            plt.figure()
            plt.plot(epoch, train, label='train')
            plt.plot(epoch, val, label='val')
            plt.legend(loc='best')
            plt.xlabel('Epoch')
            plt.ylabel(metric)
            plt.savefig('.'.join([prefix, metric.lower(), 'eps']))
            plt.close()

    def test(self, test_set, lr_block_size=(20, 20), metrics=[psnr, r2]):
        print('Test on', test_set)
        output_dir = self.test_dir / test_set
        output_dir.mkdir(exist_ok=True)

        # Evaluate metrics on each image
        rows = []
        for image_path in (data_dir / test_set).glob('*'):
            if image_path.is_dir():
                rows += [self.test_on_image(image_path, output_dir, lr_block_size=lr_block_size, metrics=metrics)]
        df = pd.DataFrame(rows)
        # Compute average metrics
        row = pd.Series()
        row['name'] = 'average'
        for col in df:
            if col != 'name':
                row[col] = df[col].mean()
        df = df.append(row, ignore_index=True)
        df.to_csv(str(self.test_dir / '{}/metrics.csv'.format(test_set)))

    def test_on_image(self, image_dir, output_dir, lr_block_size=(20, 20), metrics=[psnr, r2]):
        # Load images
        print('loading image pairs from {}'.format(image_dir))
        input_images, valid_image = load_image_pairs(image_dir, scale=self.scale)
        assert len(input_images) == 3
        name = input_images[-1].filename.name if hasattr(input_images[-1], 'filename') else ''
        print('Predict on image {}'.format(name))

        # Generate output image and measure run time
        # x_inputs的shape为四数组(数目，长度，宽度，通道数)
        x_inputs = [self.validate(img_to_array(im)) for im in input_images]
        assert x_inputs[0].shape[1] % lr_block_size[0] == 0
        assert x_inputs[0].shape[2] % lr_block_size[1] == 0
        x_train, _ = load_test_set((input_images, valid_image),
                                   lr_block_size=lr_block_size, scale=self.scale)

        model = self.compile(self.build_model(*x_train))
        if self.model_file.exists():
            model.load_weights(str(self.model_file))

        t_start = time.perf_counter()
        y_preds = model.predict(x_train, batch_size=1)  # 结果的shape为四维
        # 预测结束后进行恢复
        y_pred = np.empty(x_inputs[1].shape[-3:], dtype=np.float32)
        row_step = lr_block_size[0] * self.scale
        col_step = lr_block_size[1] * self.scale
        rows = x_inputs[0].shape[2] // lr_block_size[1]
        cols = x_inputs[0].shape[1] // lr_block_size[0]
        count = 0
        for j in range(rows):
            for i in range(cols):
                y_pred[i * row_step: (i + 1) * row_step, j * col_step: (j + 1) * col_step] = y_preds[count]
                count += 1
        assert count == rows * cols
        t_end = time.perf_counter()

        # Record metrics
        row = pd.Series()
        row['name'] = name
        row['time'] = t_end - t_start
        y_true = self.validate(img_to_array(valid_image))
        y_pred = self.validate(y_pred)
        for metric in metrics:
            row[metric.__name__] = K.eval(metric(y_true, y_pred))

        prototype = str(valid_image.filename) if hasattr(valid_image, 'filename') else None
        gdal_array.SaveArray(y_pred[0].squeeze().astype(np.int16),
                             str(output_dir / name),
                             prototype=prototype)

        return row
