import os
from torch.utils.data import Dataset as TorchDataset
import torch
import torchaudio
import numpy as np
import pandas as pd
import librosa
import math


dataset_dir = None


dataset_config = {
    'num_of_classes': 9
}


def pad_or_truncate(x, audio_length):
    """Pad all audio to specific length."""
    if len(x) <= audio_length:
        return np.concatenate((x, np.zeros(audio_length - len(x), dtype=np.float32)), axis=0)
    else:
        return x[0: audio_length]


def pydub_augment(waveform, gain_augment=0):
    if gain_augment:
        gain = torch.randint(gain_augment * 2, (1,)).item() - gain_augment
        amp = 10 ** (gain / 20)
        waveform = waveform * amp
    return waveform


class MixupDataset(TorchDataset):
    """ Mixing Up wave forms
    """

    def __init__(self, dataset, beta=2, rate=0.5):
        self.beta = beta
        self.rate = rate
        self.dataset = dataset
        print(f"Mixing up waveforms from dataset of len {len(dataset)}")

    def __getitem__(self, index):
        if torch.rand(1) < self.rate:
            x1, y1 = self.dataset[index]
            idx2 = torch.randint(len(self.dataset), (1,)).item()
            x2, y2 = self.dataset[idx2]
            l = np.random.beta(self.beta, self.beta)
            l = max(l, 1. - l)
            x1 = x1 - x1.mean()
            x2 = x2 - x2.mean()
            x = (x1 * l + x2 * (1. - l))
            x = x - x.mean()
            return x, (y1 * l + y2 * (1. - l))
        return self.dataset[index]

    def __len__(self):
        return len(self.dataset)

class MyDataset(TorchDataset):
    def __init__(self, datasource, labels, window_size, step_size=1, transform=None, resamplerate = 16000, labelrate = 10000, classes_num = 9):
        self.datasource = datasource
        self.labels = labels
        
        self.window_size = int(window_size*resamplerate)
        self.step_size = int(step_size*resamplerate)
        self.window_size_label = int(window_size*labelrate)
        self.step_size_label = int(step_size*labelrate)
        self.transform = transform
        self.resamplerate = resamplerate
        self.labelrate = labelrate
        # self.idx_inter = 0
        # self.subject_id = 0
        self.classes_num = classes_num

        self.lengths = []
        for i in range(self.datasource.shape[0]):
            cur_len =  math.floor((self.datasource[i].shape[0] - self.window_size) / self.step_size) + 1
            self.lengths.append(cur_len)
        self.lengths = np.array(self.lengths)

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self):   # avoid looping
            raise IndexError() 
        idx_inter, subject_id = self.calculate_indices(idx)
        start = idx_inter * self.step_size
        end = start + self.window_size
        start_label = idx_inter * self.step_size_label
        end_label = start_label + self.window_size_label

        waveform = self.datasource[subject_id][start:end]
        if len(waveform) == 0:
            print(idx_inter, subject_id)
        waveform_tensor = torch.from_numpy(waveform).float()

        resampler = torchaudio.transforms.Resample(orig_freq=self.resamplerate, new_freq=32000)
        data = resampler(waveform_tensor)
        
        label = self.generate_label(self.labels[subject_id][start_label:end_label,:])
        if self.classes_num == 3:
            if 2 < label < 7: label = 2 
            else: label
        target = [0] * self.classes_num
        if label < self.classes_num:
            target[label] = float(1)       
        return data.reshape(1, -1), np.array(target)
    
    def __len__(self):
        return sum(self.lengths)
    
    def generate_label(self, label):
        return np.argmax(np.sum(label,axis=0))
    
    def calculate_indices(self, idx):
        for i in range(len(self.lengths)):
            if idx < self.lengths[i]:
                break
            idx -= self.lengths[i]    
        return idx, i

def get_mixup_set(datasource, labels, window_size, step_size=1, transform=None, resamplerate = 16000, labelrate = 10000, classes_num = 9, wavmix=False):
    ds = MyDataset(datasource, labels, window_size, step_size=step_size, transform=None, resamplerate = resamplerate, labelrate = labelrate, classes_num = classes_num)
    if wavmix:
        ds = MixupDataset(ds)
    return ds

