#coding: utf-8

import os
import time
import random
import random
import torch
import torchaudio

import numpy as np
import soundfile as sf
import torch.nn.functional as F

from torch import nn
from torch.utils.data import DataLoader

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

np.random.seed(1)
random.seed(1)

SPECT_PARAMS = {
    "n_fft": 2048,
    "win_length": 1200,
    "hop_length": 300
}
MEL_PARAMS = {
    "n_mels": 80,
    "n_fft": 2048,
    "win_length": 1200,
    "hop_length": 300
}

class MelDataset(torch.utils.data.Dataset):
    def __init__(self,
                 data_list,
                 sr=24000,
                 validation=False,
                 ):

        _data_list = [l[:-1].split('|') for l in data_list]
        self.data_list = [(path_a, path_b) for path_a, path_b in _data_list]

        self.sr = sr
        self.to_melspec = torchaudio.transforms.MelSpectrogram(**MEL_PARAMS)

        self.mean, self.std = -4, 4
        self.validation = validation
        self.max_mel_length = 192

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data = self.data_list[idx]
        mel_tensor_a, mel_tensor_b = self._load_data(data)
        return mel_tensor_a, mel_tensor_b
    
    def _load_data(self, path):
        wave_tensor_a, wave_tensor_b = self._load_tensor(path)
        
        if not self.validation: # random scale for robustness
            random_scale = 0.5 + 0.5 * np.random.random()
            wave_tensor_a = random_scale * wave_tensor_a
            
        if not self.validation: # random scale for robustness
            random_scale = 0.5 + 0.5 * np.random.random()
            wave_tensor_b = random_scale * wave_tensor_b

        mel_tensor_a = self.to_melspec(wave_tensor_a)
        mel_tensor_a = (torch.log(1e-5 + mel_tensor_a) - self.mean) / self.std
        mel_length = mel_tensor_a.size(1)
        if mel_length > self.max_mel_length:
            random_start = np.random.randint(0, mel_length - self.max_mel_length)
            mel_tensor_a = mel_tensor_a[:, random_start:random_start + self.max_mel_length]
        
        mel_tensor_b = self.to_melspec(wave_tensor_b)
        mel_tensor_b = (torch.log(1e-5 + mel_tensor_b) - self.mean) / self.std
        mel_length = mel_tensor_b.size(1)
        if mel_length > self.max_mel_length:
            random_start = np.random.randint(0, mel_length - self.max_mel_length)
            mel_tensor_b = mel_tensor_b[:, random_start:random_start + self.max_mel_length]

        return mel_tensor_a, mel_tensor_b

    def _preprocess(self, wave_tensor, ):
        mel_tensor = self.to_melspec(wave_tensor)
        mel_tensor = (torch.log(1e-5 + mel_tensor) - self.mean) / self.std
        return mel_tensor

    def _load_tensor(self, data):
        path_a, path_b = data
        wave_a, sr = sf.read(path_a)
        wave_tensor_a = torch.from_numpy(wave_a).float()
        wave_b, sr = sf.read(path_b)
        wave_tensor_b = torch.from_numpy(wave_b).float()
        return wave_tensor_a, wave_tensor_b

class Collater(object):
    """
    Args:
      adaptive_batch_size (bool): if true, decrease batch size when long data comes.
    """

    def __init__(self, return_wave=False):
        self.text_pad_index = 0
        self.return_wave = return_wave
        self.max_mel_length = 192
        self.mel_length_step = 16
        self.latent_dim = 16

    def __call__(self, batch):
        batch_size = len(batch)
        nmels = batch[0][0].size(0)
        mels = torch.zeros((batch_size, nmels, self.max_mel_length)).float()
        mels2 = torch.zeros((batch_size, nmels, self.max_mel_length)).float()

        for bid, (mel, mel2) in enumerate(batch):
            mel_size = mel.size(1)
            mels[bid, :, :mel_size] = mel
            
            mel_size = mel2.size(1)
            mels2[bid, :, :mel_size] = mel2
        
        mels, mels2 = mels.unsqueeze(1), mels2.unsqueeze(1)
        return mels, mels2

def build_dataloader(path_list,
                     validation=False,
                     batch_size=4,
                     num_workers=1,
                     device='cpu',
                     collate_config={},
                     dataset_config={}):

    dataset = MelDataset(path_list, validation=validation)
    collate_fn = Collater(**collate_config)
    data_loader = DataLoader(dataset,
                             batch_size=batch_size,
                             shuffle=(not validation),
                             num_workers=num_workers,
                             drop_last=True,
                             collate_fn=collate_fn,
                             pin_memory=(device != 'cpu'))

    return data_loader
