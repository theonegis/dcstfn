from pathlib import Path
import numpy as np
from functools import partial

from keras.preprocessing.image import img_to_array
from osgeo import gdal_array
from PIL import Image

repo_dir = Path(__file__).parents[1]
data_dir = repo_dir / 'data'

input_suffix = 'input'
pred_suffix = 'pred'
valid_suffix = 'valid'

modis_prefix = 'MOD09A1'
landsat_prefix = 'LC08'


def gen_patches(image, size, stride=None):
    """将输入图像分割成给定大小的小块"""
    if not isinstance(size, tuple):
        size = (size, size)
    if stride is None:
        stride = size
    elif not isinstance(stride, tuple):
        stride = (stride, stride)
    # 这里是列优先
    for i in range(0, image.size[0] - size[0] + 1, stride[0]):
        for j in range(0, image.size[1] - size[1] + 1, stride[1]):
            yield image.crop([i, j, i + size[0], j + size[1]])


def load_image_pairs(directory, scale=16):
    """从指定目录中加载高低分辨率的图像对（包括两幅MODIS影像和两幅Landsat影像）"""
    path_list = []
    for path in Path(directory).glob('*.tif'):
        path_list.append(path)
    assert len(path_list) == 4

    for path in path_list:
        img_name = path.name
        if pred_suffix in img_name:
            modis_pred_path = path
        elif valid_suffix in img_name:
            landsat_valid_path = path
        elif input_suffix in img_name:
            if img_name.startswith(modis_prefix):
                modis_input_path = path
            elif img_name.startswith(landsat_prefix):
                landsat_input_path = path

    path_list = [modis_input_path, landsat_input_path, modis_pred_path, landsat_valid_path]
    image_list = []

    for path in path_list:
        data = gdal_array.LoadFile(str(path)).astype(np.int32)
        image = Image.fromarray(data)
        setattr(image, 'filename', path)
        image_list.append(image)

    assert image_list[0].size == image_list[0].size
    assert image_list[1].size == image_list[1].size
    assert image_list[1].size[0] == image_list[0].size[0] * scale
    assert image_list[1].size[1] == image_list[0].size[1] * scale

    return image_list[:3], image_list[-1]


def sample_to_array(samples, lr_gen_sub, hr_gen_sub, patches):
    # samples是当前批次的图片，patches是存储的容器
    assert len(samples) == 4
    for i in range(4):
        if i % 2 == 0:
            patches[i] += [img_to_array(img) for img in lr_gen_sub(samples[i])]
        else:
            patches[i] += [img_to_array(img) for img in hr_gen_sub(samples[i])]


def load_train_set(image_dir, lr_sub_size=10, lr_sub_stride=5, scale=16):
    """从给定的数据目录中加载高低分辨率的数据（根据高分辨率图像采样得到低分辨的图像）"""
    hr_sub_size = lr_sub_size * scale
    hr_sub_stride = lr_sub_stride * scale

    lr_gen_sub = partial(gen_patches, size=lr_sub_size, stride=lr_sub_stride)
    hr_gen_sub = partial(gen_patches, size=hr_sub_size, stride=hr_sub_stride)

    patches = [[] for _ in range(4)]
    for path in (data_dir / image_dir).glob('*'):
        if path.is_dir():
            print('loading image pairs from {}'.format(path))
            samples = load_image_pairs(path, scale=scale)
            samples = [*samples[0], samples[1]]
            sample_to_array(samples, lr_gen_sub, hr_gen_sub, patches)

    for i in range(4):
        patches[i] = np.stack(patches[i])
    # 返回结果为一个四维的数组(数目，长度，宽度，通道数)
    return patches[:3], patches[-1]


def load_test_set(samples, lr_block_size=(20, 20), scale=16):
    assert len(samples) == 2
    hr_block_size = [m * scale for m in lr_block_size]
    lr_gen_sub = partial(gen_patches, size=tuple(lr_block_size))
    hr_gen_sub = partial(gen_patches, size=tuple(hr_block_size))

    patches = [[] for _ in range(4)]
    samples = [*samples[0], samples[1]]
    sample_to_array(samples, lr_gen_sub, hr_gen_sub, patches)

    for i in range(4):
        patches[i] = np.stack(patches[i])
    return patches[:3], patches[-1]
