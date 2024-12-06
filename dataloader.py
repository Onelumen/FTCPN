import json
import logging
import os
import random
# import coordinate_conversion as cc
import numpy as np
import torch
import torch.utils.data as tu_data
from scipy.interpolate import interp1d

class DataGenerator:
    def __init__(self, data_path, minibatch_len, interval=1, use_preset_data_ranges=False,
                 train=True, test=True, dev=True, train_shuffle=True, test_shuffle=False, dev_shuffle=True,
                 mytest=True):

        # 检查路径是否有问题
        assert os.path.exists(data_path)
        # 规定数据属性名
        self.attr_names = ['lon', 'lat', 'alt', 'spdx', 'spdy', 'spdz']
        # 确定几个需要传入的参数数值
        self.data_path = data_path
        self.interval = interval
        self.minibatch_len = minibatch_len

        # data_status是什么作用？
        self.data_status = np.load('data_ranges.npy', allow_pickle=True).item()

        assert type(self.data_status) is dict   # 需要data_status是字典类型

        # present_data_ranges是各个数据的范围，猜测应该是用来规定各个属性的范围的
        self.preset_data_ranges = {"lon": {"max": 113.689, "min": 93.883}, "lat": {"max": 37.585, "min": 19.305},
                                           "alt": {"max": 1500, "min": 0}, "spdx": {"max": 878, "min": -945},
                                           "spdy": {"max": 925, "min": -963}, "spdz": {"max": 43, "min": -48}}
        self.use_preset_data_ranges = use_preset_data_ranges

        # 生成数据集：
        if train:
            self.train_set = mini_DataGenerator(self.readtxt(os.path.join(self.data_path, 'train'), shuffle=train_shuffle))
        if dev:
            self.dev_set = mini_DataGenerator(self.readtxt(os.path.join(self.data_path, 'dev'), shuffle=dev_shuffle))
        if test:
            self.test_set = mini_DataGenerator(self.readtxt(os.path.join(self.data_path, 'test'), shuffle=test_shuffle))
        if mytest:
            self.set=mini_DataGenerator(self.readtxt(os.path.join(self.data_path, 'mysingletest'), shuffle=test_shuffle))
            print(set)
        if use_preset_data_ranges:
            assert self.preset_data_ranges is not None

        print('data range:', self.data_status)

    def readtxt(self, data_path, shuffle=True):
        # 如果没有这个存在的路径则报错
        assert os.path.exists(data_path)
        # 创建data数据
        data = []
        for root, dirs, file_names in os.walk(data_path):
            for file_name in file_names: # 先过文件名称
                if not file_name.endswith('txt'): # 如果不是txt文件，则跳过
                    continue
                with open(os.path.join(root, file_name)) as file: # 是txt文件，打开文件
                    lines = file.readlines()
                    lines = lines[::self.interval]
                    if len(lines) == self.minibatch_len:
                        data.append(lines)
                    elif len(lines) < self.minibatch_len:
                        continue
                    else:
                        for i in range(len(lines)-self.minibatch_len+1):
                            data.append(lines[i:i+self.minibatch_len])
        print(f'{len(data)} items loaded from \'{data_path}\'')
        if shuffle:# 打乱数据顺序
            random.shuffle(data)
        return data # 返回一个数组的数据

    def scale(self, inp, attr): # 归一化处理
        assert type(attr) is str and attr in self.attr_names
        data_status = self.data_status if not self.use_preset_data_ranges else self.preset_data_ranges
        inp = (inp-data_status[attr]['min'])/(data_status[attr]['max']-data_status[attr]['min'])
        return inp

    def unscale(self, inp, attr): # 反归一化处理
        assert type(attr) is str and attr in self.attr_names
        data_status = self.data_status if not self.use_preset_data_ranges else self.preset_data_ranges
        inp = inp*(data_status[attr]['max']-data_status[attr]['min'])+data_status[attr]['min']
        return inp

    def collate(self, inp):
        '''
        :param inp: batch * n_sequence * n_attr
        :return:
        '''
        # 1. 归一化处理
        oup = []
        for minibatch in inp:
            tmp = []
            for line in minibatch:
                items = line.strip().split("|")
                lon, lat, alt, spdx, spdy, spdz = float(items[4]), float(items[5]), int(float(items[6]) / 10), \
                                                  float(items[7]), float(items[8]), float(items[9])
                tmp.append([lon, lat, alt, spdx, spdy, spdz])
            minibatch = np.array(tmp)
            for i in range(minibatch.shape[-1]):
                minibatch[:, i] = self.scale(minibatch[:, i], self.attr_names[i])
            oup.append(minibatch)
        # return np.array(oup)
        batch_data = np.array(oup)

        #   2. 掩蔽数据

        def mask_data(data):
            mask = np.ones_like(data) # 遮蔽矩阵
            indexlist=[]
            for i in range(data.shape[0]):
                rand_num = random.randint(0, data.shape[1] - 1)
                mask[i][rand_num] = float('nan')  # 用 NaN 表示掩蔽值
                indexlist.append(rand_num)
            return indexlist,mask


        list,mask = mask_data(batch_data)
        masked_data = batch_data * mask

        def interpolate_missing_data(batch):
            batch_interpolated = []
            for sequence in batch:
                interpolated_sequence = []
                for i in range(sequence.shape[1]):
                    col = sequence[:, i]  # 获取第i个属性的列 (T, )
                    valid_idx = ~np.isnan(col)
                    valid_data = col[valid_idx]
                    time_idx = np.arange(len(col))
                    valid_time_idx = time_idx[valid_idx]

                    # 插值
                    if len(valid_time_idx) > 1:
                        interp_func = interp1d(valid_time_idx, valid_data, kind='linear', fill_value="extrapolate")
                        interpolated_col = interp_func(time_idx)
                    else:
                        interpolated_col = col

                    interpolated_sequence.append(interpolated_col)
                batch_interpolated.append(np.stack(interpolated_sequence, axis=-1))  # 重新组合列
            return np.stack(batch_interpolated)
        interpolated_batch = interpolate_missing_data(masked_data)

        return {
            'original': torch.tensor(batch_data, dtype=torch.float32),
            'masked': torch.tensor(masked_data, dtype=torch.float32),
            'indexlist':list,
            'interpolated': torch.tensor(interpolated_batch, dtype=torch.float32)
        }

class mini_DataGenerator(tu_data.Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)

