from pathlib import Path
import numpy as np
import rasterio
from PIL import Image
import math
from collections import OrderedDict

import torch
from torch.utils.data import Dataset
from utils import make_tuple

root_dir = Path(__file__).parents[1]
data_dir = root_dir / 'data'

reference = '00'
predict_prefix = '01'

coarse_prefix = 'MOD09A1'
fine_prefix = 'LC08'


def load_image_pair(im_dir, scale):
    """
    从指定目录中加载一组高低分辨率的图像对
    """
    # 按照一定顺序获取给定文件夹下的一组数据
    paths = get_image_pair(im_dir)

    # 将组织好的数据转为Image对象
    images = []
    for p in paths:
        with rasterio.open(str(p)) as ds:
            im = ds.read().astype(np.float32)   # CHW
            im = Image.fromarray(im[0])         # HW
            images.append(im)

    # 对数据的尺寸进行验证（等于4的时候是训练数据集，等于3的时候是实际预测）
    assert len(images) == 4 or len(images) == 3
    assert (images[0].size[0] * scale, images[0].size[1] * scale) == images[1].size
    # 返回训练数据和验证数据（最后一个是验证数据）
    return images


def get_image_pair(im_dir):
    # 在该实验中，所有的图像都按照如下顺序进行组织
    order = OrderedDict()
    order[0] = reference + '_' + coarse_prefix
    order[1] = reference + '_' + fine_prefix
    order[2] = predict_prefix + '_' + coarse_prefix
    order[3] = predict_prefix + '_' + fine_prefix
    # paths用于存储组织好的数据对
    paths = []

    # 将一组数据集按照规定的顺序组织好
    for label in order.values():
        for p in Path(im_dir).glob('*.tif'):
            if p.name.startswith(label):
                paths.append(p.expanduser().resolve())
                break
    return paths


class PatchSet(Dataset):
    """
    每张图片分割成小块进行加载
    参考影像个数只支持1和2
    """

    def __init__(self, image_dir, image_size, patch_size, patch_stride=None, scale=16):
        super(PatchSet, self).__init__()
        patch_size = make_tuple(patch_size)
        if not patch_stride:
            patch_stride = patch_size
        else:
            patch_stride = make_tuple(patch_stride)

        self.root_dir = image_dir
        self.image_size = image_size
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.scale = scale

        self.image_dirs = [p for p in self.root_dir.glob('*') if p.is_dir()]
        self.num_im_pairs = len(self.image_dirs)

        # 计算出图像进行分块以后的patches的数目
        self.num_patches_x = math.ceil((image_size[0] - patch_size[0] + 1) / patch_stride[0])
        self.num_patches_y = math.ceil((image_size[1] - patch_size[1] + 1) / patch_stride[1])
        self.num_patches = self.num_im_pairs * self.num_patches_x * self.num_patches_y

    @staticmethod
    def im2tensor(im):
        size = im.size  # WH
        im = np.array(im, np.float32, copy=False) * 0.0001
        im = torch.from_numpy(im)
        im = im.view(size[1], size[0], 1)
        im = im.transpose(0, 1).transpose(0, 2).contiguous()  # From HWC to CHW
        return im

    def __getitem__(self, index):
        id_n, id_x, id_y = self.map_index(index)
        images = load_image_pair(self.image_dirs[id_n], self.scale)
        patches = [None] * len(images)

        for i in range(len(patches)):
            if i % 2:
                im = images[i].crop([id_x * self.scale, id_y * self.scale,
                                     (id_x + self.patch_size[0]) * self.scale,
                                     (id_y + self.patch_size[1]) * self.scale])
            else:
                im = images[i].crop([id_x, id_y,
                                     id_x + self.patch_size[0],
                                     id_y + self.patch_size[1]])
            patches[i] = self.im2tensor(im)

        del images[:]
        del images

        if len(patches) == 3:
            return patches, None
        return patches[:-1], patches[-1]

    def __len__(self):
        return self.num_patches

    def map_index(self, index):
        # 将全局的index映射到具体的图像对文件夹索引(id_n)，图像裁剪的列号与行号(id_x, id_y)
        id_n = index // (self.num_patches_x * self.num_patches_y)
        residual = index % (self.num_patches_x * self.num_patches_y)
        id_x = self.patch_stride[0] * (residual % self.num_patches_x)
        id_y = self.patch_stride[1] * (residual // self.num_patches_x)
        return id_n, id_x, id_y
