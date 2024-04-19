import os
from torch.utils.data import Dataset as TorchDataset
import torch
import torchaudio
import numpy as np
import pandas as pd
import librosa
import math
import scipy


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
    def __init__(self, datasource, labels, window_size, step_size=1, transform=None, resamplerate = 16000, 
                 labelrate = 10000, classes_num = 9, acc_rate = 98.6, gyro_rate = 98.6, mag_rate = 25):
        self.datasource = datasource[0]
        self.labels = labels
        self.all_acc = datasource[1]
        self.all_gyro = datasource[2]
        self.all_mag = datasource[3]
        
        self.window_size, self.step_size = int(window_size*resamplerate), int(step_size*resamplerate)
        self.window_size_label, self.step_size_label = int(window_size*labelrate), int(step_size*labelrate)
        self.window_size_acc, self.step_size_acc = int(window_size*acc_rate), int(step_size*acc_rate)
        self.window_size_gyro, self.step_size_gyro = int(window_size*gyro_rate), int(step_size*gyro_rate)
        self.window_size_mag, self.step_size_mag = int(window_size*mag_rate), int(step_size*mag_rate)

        self.transform = transform
        self.resamplerate = resamplerate
        self.magupsample = round(acc_rate/mag_rate) # This need to be rewritten if IMU sampling rates change
        self.labelrate = labelrate
        self.classes_num = classes_num

        self.lengths = []
        for i in range(self.datasource.shape[0]):
            cur_len =  math.floor((self.datasource[i].shape[0] - self.window_size) / self.step_size) + 1
            self.lengths.append(cur_len)
        self.lengths = np.array(self.lengths)

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self):  
            raise IndexError() 
        idx_inter, subject_id = self.calculate_indices(idx)
        start, end = self.start_end_indices(idx_inter, self.step_size, self.window_size)
        start_label, end_label = self.start_end_indices(idx_inter, self.step_size_label, self.window_size_label)
        start_acc, end_acc = self.start_end_indices(idx_inter, self.step_size_acc, self.window_size_acc)
        start_gyro, end_gyro = self.start_end_indices(idx_inter, self.step_size_gyro, self.window_size_gyro)
        start_mag, end_mag = self.start_end_indices(idx_inter, self.step_size_mag, self.window_size_mag)

        ## Audio 
        waveform = self.datasource[subject_id][start:end]
        if len(waveform) == 0:
            print(idx_inter, subject_id)
        waveform_tensor = torch.from_numpy(waveform).float()

        resampler = torchaudio.transforms.Resample(orig_freq=self.resamplerate, new_freq=32000)
        data = resampler(waveform_tensor)
        
        ## IMU
        accelerometer = self.all_acc[subject_id][:,start_acc:end_acc]
        gyroscope = self.all_gyro[subject_id][:,start_gyro:end_gyro]
        magnetometer = self.all_mag[subject_id][:,start_mag:end_mag]
        magnetometer = np.repeat(magnetometer, self.magupsample, axis=1)
        # min_length = min(accelerometer.shape[1], gyroscope.shape[1], magnetometer.shape[1])
        IMU = np.stack((self.fix_array_length(accelerometer), self.fix_array_length(gyroscope), self.fix_array_length(magnetometer)), axis = 0)

        ## Label
        label = self.generate_label(self.labels[subject_id][start_label:end_label,:])
        if self.classes_num == 3:
            if 2 < label < 7: label = 2 
            else: label
        target = [0] * self.classes_num
        if label < self.classes_num:
            target[label] = float(1)  
                
        return data.reshape(1, -1), IMU, np.array(target)
    
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
    
    def start_end_indices(self, iter_idx, step_size, window_size):
        start = iter_idx * step_size
        end = start + window_size
        return start, end
    
    def fix_array_length(self, arr, target_length = 147):
        """
        Adjusts an array's length to the target length.
        Truncates if the array is longer; repeats the last element if shorter.

        :param arr: Input array.
        :param target_length: Desired length of the array.
        :return: Adjusted array.
        """
        current_length = arr.shape[-1] 
        if current_length > target_length:
            # Truncate the array if it's too long
            return arr[:,:target_length]
        elif current_length < target_length:
            # Extend the array by repeating the last element if it's too short
            padding = (0, target_length - current_length)
            return np.pad(arr, pad_width=((0, 0), padding), mode='edge')
        else:
            # Return the array as is if it's already the correct length
            return arr


def get_mixup_set(datasource, labels, window_size, step_size=1, transform=None, resamplerate = 16000, labelrate = 10000, classes_num = 9, wavmix=False):
    ds = MyDataset(datasource, labels, window_size, step_size=step_size, transform=None, resamplerate = resamplerate, labelrate = labelrate, classes_num = classes_num)
    if wavmix:
        ds = MixupDataset(ds)
    return ds

