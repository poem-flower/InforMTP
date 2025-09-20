import os
import numpy as np
import pandas as pd
import scipy.io as scio

import torch
from sympy.physics.control.tests.test_control_plots import numpy
from torch.utils.data import Dataset, DataLoader

from utils.tools import StandardScaler
from utils.timefeatures import time_features

import warnings
warnings.filterwarnings('ignore')


class My_Dataset(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='my_data.mat',
                 target='OT', scale=False, inverse=False, timeenc=0, freq='h', cols=None, IsDoppler=0):
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]

        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        #self.scaler = StandardScaler()
        data_input = np.array(scio.loadmat(self.root_path + self.data_path)['Input_data']) 
        data_true = np.array(scio.loadmat(self.root_path + self.data_path)['True_data']) 
        data_pred = np.array(scio.loadmat(self.root_path + self.data_path)['Pred_data'])

        self.data_x = data_input
        self.data_true = data_true
        self.data_pred = data_pred

        self.M = self.data_x.shape[0]
        self.N = self.data_x.shape[1]
        self.n = self.N - self.seq_len - self.pred_len + 1

    def __getitem__(self, index):
        i = index // self.n
        j = index % self.n

        seq_x = self.data_x[i, j:j+self.seq_len, :]
        dec_padding = np.zeros((self.pred_len, seq_x.shape[-1]))
        seq_y = np.concatenate((seq_x, dec_padding), axis=0)
        seq_true = np.concatenate([self.data_true[i, j:j+self.seq_len], self.data_pred[i, j+self.seq_len:j+self.seq_len+self.pred_len]], 0) #输出序列，loss计算


        return seq_x, seq_y, seq_true

    def __len__(self):
        return self.M * self.n

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)



