import keras.layers
from keras.layers import Input, Conv2D, Conv2DTranspose, MaxPooling2D, Dense
from keras.models import Model, Sequential

from toolbox.misc import factorize


def dcfn(coarse_input, fine_input, coarse_pred, d=[64, 96, 128]):
    # 对于两张MODIS影像建立相同的网络
    assert coarse_input.shape == coarse_pred.shape

    coarse_model = Sequential()
    coarse_model.add(Conv2D(d[0], 3, padding='same',
                            kernel_initializer='he_normal',
                            activation='relu', input_shape=coarse_input.shape[-3:]))
    coarse_model.add(Conv2D(d[1], 3, padding='same',
                            kernel_initializer='he_normal',
                            activation='relu'))

    factors = factorize(16)
    for n in factors:
        coarse_model.add(Conv2DTranspose(d[1], 3, strides=n, padding='same',
                                         kernel_initializer='he_normal'))
        coarse_model.add(Conv2D(d[1], 3, padding='same',
                                kernel_initializer='he_normal',
                                activation='relu'))

    # 对于Landsat高分辨率影像建立网络
    pool_size = 2
    fine_model = Sequential()
    fine_model.add(Conv2D(d[0], 3, padding='same',
                          kernel_initializer='he_normal',
                          activation='relu', input_shape=fine_input.shape[-3:]))
    fine_model.add(Conv2D(d[1], 3, padding='same',
                          kernel_initializer='he_normal',
                          activation='relu'))
    fine_model.add(MaxPooling2D(pool_size=pool_size, padding='same'))
    fine_model.add(Conv2D(d[1], 3, padding='same',
                          kernel_initializer='he_normal',
                          activation='relu'))
    fine_model.add(Conv2D(d[2], 3, padding='same',
                          kernel_initializer='he_normal',
                          activation='relu'))
    fine_model.add(Conv2DTranspose(d[1], 3, strides=pool_size, padding='same',
                                   kernel_initializer='he_normal'))

    # 三个网络的融合
    coarse_input_layer = Input(shape=coarse_input.shape[-3:])
    coarse_input_model = coarse_model(coarse_input_layer)
    fine_input_layer = Input(shape=fine_input.shape[-3:])
    fine_input_model = fine_model(fine_input_layer)
    subtracted_layer = keras.layers.subtract([fine_input_model, coarse_input_model])
    coarse_pred_layer = Input(shape=coarse_pred.shape[-3:])
    coarse_pred_model = coarse_model(coarse_pred_layer)
    added_layer = keras.layers.add([subtracted_layer, coarse_pred_model])
    final_out = Dense(fine_input.shape[-1])(added_layer)
    model = Model([coarse_input_layer, fine_input_layer, coarse_pred_layer], final_out)
    return model


def get_model(name):
    """通过字符串形式的函数名称得到该函数对象，可以直接对该函数进行调用"""
    return globals()[name]
