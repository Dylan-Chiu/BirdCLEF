import pandas as pd
import data_utils
import torch
import time
import sys
import os
from datetime import datetime
import random
import torchvision.transforms as transforms
import my_transform
import multiprocessing
from tqdm import tqdm
from functools import partial




class BirdDataSet:
    def __init__(self, csv_path, audio_path, fill_length, rating_min,
                 use_premphasis, use_remove_silence, top_db, use_cache, use_location, num_load_processes,
                 transform=None):  # train模式下transform不为none
        # 初始数据
        self.audio_path = audio_path
        self.audio_pt_path = os.path.splitext(csv_path)[0] + ".pt"

        self.metadata_df = pd.read_csv(csv_path)
        self.transform = None
        self.fill_length = fill_length

        self.use_premphasis = use_premphasis
        self.use_remove_silence = use_remove_silence
        self.use_location = use_location
        self.top_db = top_db
        self.num_load_processes = num_load_processes

        # 删除坐标值为空的行，指定检查longitude和latitude列
        src_num = len(self.metadata_df)  # 原本有几个样本
        # 删除评级过低的值
        self.metadata_df = self.metadata_df[self.metadata_df['rating'] >= rating_min]
        self.metadata_df = self.metadata_df.dropna(subset=['longitude', 'latitude']).reset_index(drop=True)

        # 类别编码 (将'Class_1'这样的字符串统一编码为数字，并用两个字典进行转化)
        classes = self.metadata_df.loc[:, 'primary_label'].unique()  # 拿到所有类别的不重复的名称
        self.classes_num = len(classes)
        self.idx_to_class = {i: x for i, x in enumerate(classes)}  # 生成{0: 'Class_1' ...}的字典
        self.class_to_idx = {x: i for i, x in self.idx_to_class.items()}  # 生成{'Class_1': 0 ...}的字典

        # 初始化时将数据全部读入并转化后写入内存以提高运行速度（全部读入内存的写法）
        print(f"Number of classes: {self.classes_num}")
        print(f"Number of audios: {len(self.metadata_df)} (src: {src_num})")
        start_time = time.time()
        if os.path.exists(self.audio_pt_path) and use_cache:  # 文件存在，并且使用缓存
            print(f"'{os.path.basename(self.audio_pt_path)}' is exists!")
            print("loading...")
            self.audio_data = torch.load(self.audio_pt_path)
        else:
            print(f"'{os.path.basename(self.audio_pt_path)}' is not exists!")
            self.audio_data = self._load_all_data_parallel(self.metadata_df, fill_length, self.num_load_processes)
            print("saving...")
            torch.save(self.audio_data, self.audio_pt_path)
        end_time = time.time()
        running_time = end_time - start_time
        print(f"Running time: {running_time:.2f} s")
        print(f"Size of audio_data: {os.path.getsize(self.audio_pt_path) / 1000 / 1000 / 1000:.2f} GB")  # 这里计算有问题！

    def __getitem__(self, index):  # （全部读入内存的写法）
        label, filename, latitude, longitude = self.metadata_df.loc[
            index, ['primary_label', 'filename', 'latitude', 'longitude']]
        audio = self.audio_data[index]
        location_packet = torch.tensor([longitude, latitude], dtype=torch.float32)
        if self.transform is not None:  # train
            audio = self.transform(audio)  # train DataSet 将会在此处被切割为符合规范的尺寸
        return audio, location_packet, self.class_to_idx[label]

    # def __getitem__(self, index):
    #     label, filename, latitude, longitude = self.metadata_df.loc[
    #             index, ['primary_label', 'filename', 'latitude', 'longitude']]
    #     audio = self._load_single_data(self.metadata_df, self.fill_length, index)
    #     if self.transform is not None:  # train
    #         audio = self.transform(audio)  # train DataSet 将会在此处被切割为符合规范的尺寸
    #         # coords在此处被转化为网格
    #         H = audio.shape[1]
    #         W = audio.shape[2]
    #         location_matrix = data_utils.get_location_matrix(longitude, latitude, H, W,
    #                                                          self.lon_max, self.lon_min, self.lat_max, self.lat_min)
    #         data = torch.cat([audio, location_matrix], dim=0)
    #         return data, self.class_to_idx[label]
    #     else:  # valid
    #         location_packet = (longitude, latitude, self.lon_max, self.lon_min, self.lat_max, self.lat_min)
    #         return audio, location_packet, self.class_to_idx[label]

    def __len__(self):
        return len(self.metadata_df)

    def set_transform(self, transform):
        self.transform = transform

    def get_transform(self):
        return self.transform

    # 读入df所需要的音频数据，并且转化为所需tensor后，写入内存
    def _load_all_data(self, metadata_df, fill_length):
        filenames = metadata_df.loc[:, 'filename']
        bird_name = list(metadata_df.loc[:, 'primary_label'])
        tensor_list = []
        for index, filename in tqdm(enumerate(filenames), total=len(filenames)):
            if '/' not in filename:
                filename = bird_name[index] + '/' + filename
            mel = data_utils.get_mel_from_audio(self.audio_path + filename, fill_length,
                                                self.use_premphasis, self.use_remove_silence, self.top_db)
            tensor_list.append(mel)
        return tensor_list

    def _load_all_data_parallel(self, metadata_df, fill_length, num_processes):
        filenames = metadata_df.loc[:, 'filename']
        bird_name = list(metadata_df.loc[:, 'primary_label'])
        filepaths = []
        for index, filename in enumerate(filenames):
            if '/' not in filename:
                filename = bird_name[index] + '/' + filename
            filepaths.append(self.audio_path + filename)
        pool = multiprocessing.Pool(num_processes)
        get_mel = partial(data_utils.get_mel_from_audio,
                          fill_length=fill_length,
                          use_premphasis=self.use_premphasis,
                          use_remove_silence=self.use_remove_silence,
                          top_db=self.top_db)
        tensor_list = []
        for result in tqdm(pool.imap_unordered(get_mel, filepaths), total=len(filepaths)):
            tensor_list.append(result)
        return tensor_list


    # 随机找一个频谱图，再随机裁剪一块出来
    def random_patch(self, crop_len):
        num_audio = len(self.audio_data)
        rand_idx = random.randint(0, num_audio - 1)
        rand_audio = self.audio_data[rand_idx]
        trans = my_transform.RandomHorizontalCrop(crop_len)
        rand_patch = trans(rand_audio)
        return rand_patch


# 测试
if __name__ == '__main__':
    pass
