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

modis_prefix = 'MYD09A1'
landsat_prefix = 'LC8'


def mod_crop(image, scale):
    """使得图像的大小是scale的整数倍，方便后续处理"""
    size = np.array(image.size)
    size -= size % scale
    return image.crop([0, 0, *size])


def gen_patches(image, size, stride):
    """将输入图像分割成给定大小的小块"""
    for i in range(0, image.size[0] - size + 1, stride):
        for j in range(0, image.size[1] - size + 1, stride):
            yield image.crop([i, j, i + size, j + size])


def load_pairs(directory, scale=16):
    """从指定目录中加载高低分辨率的图像对（包括两幅MODIS影像和两幅Landsat影像）"""
    path_list = []
    for path in Path(directory).glob('*'):
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


def load_set(image_dir, lr_sub_size=10, lr_sub_stride=5, scale=16):
    """从给定的数据目录中加载高低分辨率的数据（根据高分辨率图像采样得到低分辨的图像）"""
    hr_sub_size = lr_sub_size * scale
    hr_sub_stride = lr_sub_stride * scale

    lr_gen_sub = partial(gen_patches, size=lr_sub_size, stride=lr_sub_stride)
    hr_gen_sub = partial(gen_patches, size=hr_sub_size, stride=hr_sub_stride)

    sample_patches = [[], [], [], []]
    for path in (data_dir / image_dir).glob('*'):
        if path.is_dir():
            samples = load_pairs(path, scale=scale)
            samples = [samples[0][0], samples[0][1], samples[0][2], samples[1]]
            for i in range(4):
                if i % 2 == 0:
                    sample_patches[i] += [img_to_array(img) for img in lr_gen_sub(samples[i])]
                else:
                    sample_patches[i] += [img_to_array(img) for img in hr_gen_sub(samples[i])]

    for i in range(4):
        sample_patches[i] = np.stack(sample_patches[i])
    # 返回结果为一个四维的数组(数目，长度，宽度，通道数)
    return sample_patches[:3], sample_patches[-1]
