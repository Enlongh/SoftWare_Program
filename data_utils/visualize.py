from pathlib import Path

import nibabel as nib
import numpy as np
from src.config import config


def load_img(path, Window_min=None, Window_max=None):
    """
    :param path: 绝对路径
    :return: numpy矩阵，形状为[D,H,W]，归一化之后的img
    """
    img = nib.load(str(path)).get_fdata().astype(np.float32)

    if Window_min != None:
        config.CutWindow[0] = Window_min
    if Window_max != None:
        config.CutWindow[1] = Window_max

    img[img < config.CutWindow[0]] = config.CutWindow[0]
    img[img > config.CutWindow[1]] = config.CutWindow[1]
    Z_Score_mean = config.CutWindow[0]
    Z_Score_std = config.CutWindow[1] - config.CutWindow[0]
    img = (img - Z_Score_mean) \
          / Z_Score_std
    img = img*255
    return img
