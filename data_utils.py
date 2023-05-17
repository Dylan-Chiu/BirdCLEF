import librosa
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import SubsetRandomSampler
from torch.utils.data import DataLoader, random_split, Subset
import bird_dataset
import torchvision.transforms as transforms
import my_transform
import copy


SAMPLING_RATE = 32000  # 采样率
MAX_TIME = 1 * 60  # 音频最长长度秒数

def get_mel_from_audio(filepath, fill_length, use_premphasis, use_remove_silence, top_db):
    # 读取音频文件
    y, sr = librosa.load(filepath, sr=SAMPLING_RATE, duration=MAX_TIME)
    # 论文看的参数
    N_FFT = 2048  # FFT点数
    HOP_LENGTH = 512  # 帧移量
    N_MELS = 224  # Mel滤波器数量
    FMIN = 0  # 最低频率
    FMAX = 16000  # 最高频率
    PADDING_NUM = -80  # 长度不足时填充的值

    # 预加重
    if use_premphasis:
        y = librosa.effects.preemphasis(y)

    # 删除静音片段
    if use_remove_silence:
        y = remove_silence(y, top_db)
    s = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS, fmin=FMIN,
                                       fmax=FMAX)
    s_db = librosa.power_to_db(s, ref=np.max)
    # 对于time_steps，不足则填充
    if s_db.shape[1] < fill_length:
        s_db = np.pad(s_db, ((0, 0), (0, fill_length - s_db.shape[1])), 'constant',
                      constant_values=(PADDING_NUM, PADDING_NUM))
    # 转为tensor
    s_tensor = torch.from_numpy(s_db)
    # 增加第一个channel维度
    s_tensor = s_tensor.view(1, s_tensor.shape[0], s_tensor.shape[1])
    return s_tensor

def get_mfcc_from_audio(filepath, fill_length):
    # 读取音频文件
    y, sr = librosa.load(filepath, sr=SAMPLING_RATE, duration=MAX_TIME)
    # 论文看的参数
    N_MFCC = 128
    PADDING_NUM = 0  # 长度不足时填充的值
    s = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)

    # 对于time_steps，不足则填充
    if s.shape[1] < fill_length:
        s = np.pad(s, ((0, 0), (0, fill_length - s.shape[1])), 'constant',
                      constant_values=(PADDING_NUM, PADDING_NUM))

    # 转为tensor
    s_tensor = torch.from_numpy(s)
    # 增加第一个channel维度
    s_tensor = s_tensor.view(1, s_tensor.shape[0], s_tensor.shape[1])
    return s_tensor


def remove_silence(audio, top_db):
    clips = librosa.effects.split(audio, top_db=top_db)
    wav_data = []
    for c in clips:
        data = audio[c[0]: c[1]]
        wav_data.extend(data)
    return np.hstack(wav_data)


def show_spectrogram(s_tensor, title=None):
    """
    显示光谱图

    Args:
        s_tensor: 经过处理的tensor类型音频数据
        title: 标题

    """
    s_db = s_tensor.view(s_tensor.shape[1], s_tensor.shape[2]).numpy()
    plt.figure(figsize=(8, 2))
    librosa.display.specshow(s_db, sr=SAMPLING_RATE, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.xlabel('时间（s）', fontsize=12)
    plt.ylabel('频率（Hz）', fontsize=12)
    plt.title(title, fontsize=12)
    plt.savefig('temp.png', dpi=500, bbox_inches='tight') # 解决图片不清晰，不完整的问题
    plt.show()


def show_waveform(y, title=None, sr=32000, width_alpha=1):
    plt.figure(figsize=(8*width_alpha, 2))
    t = librosa.frames_to_time(range(len(y)), sr=sr)
    plt.plot(t, y)
    plt.xlabel('时间 (s)', fontsize=12)
    plt.ylabel('振幅', fontsize=12)
    plt.title(title, fontsize=12)
    plt.savefig('temp.png', dpi=500, bbox_inches='tight') # 解决图片不清晰，不完整的问题
    plt.show()

def get_sampler(length, valid_size):
    """
    获取采样器

    Args:
        length: 样本个数
        valid_size: 验证集比例

    Returns:
        train_sampler, valid_sampler
    """
    indices = list(range(length))
    np.random.shuffle(indices)
    split = int(length * valid_size)
    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)
    return train_sampler, valid_sampler


# 构造train, valid, 两个dataloader
def get_dataloader(dataset, valid_size, batch_size):
    """
    获取dataloader

    Args:
        dataset: 数据集
        valid_size: 验证集比例
        batch_size: 批次大小

    Returns:
        train_loader, valid_loader
    """
    train_sampler, valid_sampler = get_sampler(len(dataset), valid_size)
    train_loader = DataLoader(dataset=dataset, sampler=train_sampler, batch_size=batch_size)
    valid_loader = DataLoader(dataset=dataset, sampler=valid_sampler, batch_size=batch_size)
    return train_loader, valid_loader


def crop_with_rule(s, crop_len, crop_step, crop_max_num):
    """
    按照规则裁剪
    Args:
        s: tensor(channels, frequencies, time_steps)
        crop_len: 每个片的长度
        crop_step: 每次进行分割的步长
        crop_max_num: 最大分片数量
    Returns:
        train_loader, valid_loader
    """
    s_len = s.shape[2]
    seg_list = []
    start = 0
    while start < s_len and len(seg_list) < crop_max_num:  # 分片在界内 且 数量小于分片最大值
        # print(len(seg_list))
        end = crop_len + start
        if end >= s_len:  # 最后一片的结束为止如果超出范围
            end = s_len - 1
        seg = s[:, :, start:end]
        seg = torch.nn.functional.pad(seg, (0, crop_len - seg.shape[2]), value=-80)  # 长度不足则填充
        seg_list.append(seg)
        start += crop_step
    stacked_tensor = torch.stack(seg_list, dim=0)
    return stacked_tensor


def add_location_channel(stacked_tensor, location_packet):
    """
    添加一个坐标channel
    Args:
        stacked_tensor: tensor(batch, channels, frequencies, time_steps)([2, 1, 128, 1024])
        location_packet: (longitude, latitude, lon_max, lon_min, lat_max, lat_min)
    Returns:
        stacked_tensor_location: tensor(batch, channels, frequencies, time_steps)([2, 2, 128, 1024])
    """

    H = stacked_tensor.shape[2]
    W = stacked_tensor.shape[3]
    batch_num = stacked_tensor.shape[0]
    longitude, latitude, lon_max, lon_min, lat_max, lat_min = location_packet
    location_matrix = get_location_matrix(longitude, latitude, H, W,
                                          lon_max, lon_min, lat_max, lat_min)  # torch.Size([1, 128, 1024])
    location_matrix_repeat = location_matrix.repeat(batch_num, 1, 1, 1)  # torch.Size([2, 1, 128, 1024])
    # print(location_matrix_repeat.shape)
    # print(stacked_tensor.shape)
    stacked_tensor_location = torch.cat([stacked_tensor, location_matrix_repeat], dim=1)
    return stacked_tensor_location


# 获得train的dataloader和valid的dataset
# 因为valid不定长，不可使用dataloader
def get_train_loader_and_valid_dataset(dataset, valid_size, batch_size, train_transform):
    valid_num = int(len(dataset) * valid_size)
    indices = list(range(len(dataset)))
    np.random.shuffle(indices)
    train_indices = indices[valid_num:]
    valid_indices = indices[:valid_num]

    # 创建两个Subset对象，每个对象都有不同的索引列表和转换操作
    train_dataset = Subset(copy.copy(dataset), train_indices)
    train_dataset.dataset.transform = train_transform
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

    valid_dataset = Subset(copy.copy(dataset), valid_indices)
    return train_loader, valid_dataset


def get_location_matrix(lon, lat, H, W, lon_max, lon_min, lat_max, lat_min):
    """
    将经度和维度（两个标量）转化为一个H*W的二维矩阵，其中经纬度所在点标为1，其余点为0
    Returns:
        matrix: tensor(channels, H, W)([1, 128, 1024])
    """
    # 归一化lon和lat
    lon = (lon - lon_min) / (lon_max - lon_min)
    lat = (lat - lat_min) / (lat_max - lat_min)
    # 计算矩阵上的坐标
    x = int(lon * W)  # W较大，经度划分更精细一点
    y = int(lat * H)
    # 处于最大值的那个经纬度放置网格的时候会越界，处理一下
    if x == W:
        x -= 1
    if y == H:
        y -= 1
    # 构造矩阵
    matrix = torch.zeros((H, W))
    matrix[y, x] = 1
    matrix = matrix.unsqueeze(0)  # 前面加一个channel维度
    return matrix


def get_location_code(lon, lat, max_len, lon_max=180, lon_min=-180, lat_max=90, lat_min=-90):
    # 将经纬度归一化到 [0, 1] 的范围内
    norm_lon = (lon - lon_min) / (lon_max - lon_min)
    norm_lat = (lat - lat_min) / (lat_max - lat_min)

    # 将归一化后的经纬度缩放到 [0, max_len) 的整数范围内
    code_lon = (norm_lon * max_len).type(torch.int32)
    code_lat = (norm_lat * max_len).type(torch.int32)

    # 创建一个大小为 max_len 的数组，将编码后的经纬度位置置为 1，其余位置置为 0
    code = torch.zeros((len(lon), max_len), dtype=torch.float32)
    code[torch.arange(len(lon)), code_lon // 2] = 1
    code[torch.arange(len(lat)), max_len // 2 + code_lat // 2] = 1
    return code


# 测试
if __name__ == '__main__':
    pass
    # print(get_location_code(torch.tensor([0, -179, +179]),
    #                         torch.tensor([0, -89, +89]), 10))