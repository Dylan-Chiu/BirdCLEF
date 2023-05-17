import torchaudio
import torch
from random import randint, uniform
import random
import numpy as np
import bird_dataset
import torchvision.transforms as transforms
import data_utils


# 随机裁剪出固定大小的一块
class RandomHorizontalCrop:
    def __init__(self, crop_len):
        self.crop_len = crop_len

    def __call__(self, s):
        s_len = s.shape[2]
        start_step = random.randint(0, s_len - self.crop_len)  # 有bug
        end_step = self.crop_len + start_step
        return s[:, :, start_step:end_step]


class GaussianNoise:
    def __init__(self, std, mean=0):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        noise = torch.normal(mean=self.mean, std=self.std, size=data.shape)
        return data + noise


class TanhDistortion(object):
    def __init__(self, distortion_factor, use_tanh):
        self.distortion_factor = distortion_factor
        self.use_tanh = use_tanh

    def __call__(self, spec):
        if self.use_tanh:
            return torch.tanh(self.distortion_factor * spec)
        else:
            return spec


class MixUp(object):
    def __init__(self, dataset, ratio):
        """
        混合随机其他光谱图
        Args:
            dataset: 从dataset里面随机取出数据
            ratio: 随机的片段占新片段的比例
        """
        self.dataset = dataset
        self.ratio = ratio
    
    def __call__(self, spec):
        spec_len = spec.shape[2]
        rand_spec = self.dataset.random_patch(spec_len)
        new_spec = spec * (1-self.ratio) + rand_spec * self.ratio
        return new_spec


class Normalize(object):
    def __init__(self, use_norm):
        self.use_norm = use_norm

    def __call__(self, tensor):
        if not self.use_norm:
            return tensor
        norm = torch.norm(tensor)
        if norm == 0:
            return tensor
        return tensor / norm

# 测试
if __name__ == '__main__':
    # my_tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    # transform = Normalize(False)
    # normalized_tensor = transform(my_tensor)
    # print(normalized_tensor)
    pass
