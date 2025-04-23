# !usr/bin/env python
# -*- coding:utf-8 _*-
"""
@Author: Huiqiang Xie
@File: EurDataset.py
@Time: 2021/3/31 23:20
"""

import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset

class EurDataset(Dataset):
    def __init__(self, split='train'):
        data_dir = '/content/data/txt/'
        with open(data_dir + 'europarl/{}_data.pkl'.format(split), 'rb') as f:
            self.data = pickle.load(f)


    def __getitem__(self, index):
        sents = self.data[index]
        return  sents

    def __len__(self):
        return len(self.data)

def collate_data(batch):
    batch_size = len(batch)
    # Sort sequences by length in descending order and keep track of original indices
    sort_by_len_with_idx = sorted(enumerate(batch), key=lambda x: len(x[1]), reverse=True)
    original_idx, sort_by_len = zip(*sort_by_len_with_idx)
    
    max_len = min(32, len(sort_by_len[0]))  # Cap maximum length at 32
    sents = np.zeros((batch_size, max_len), dtype=np.int64)

    for i, sent in enumerate(sort_by_len):
        length = min(len(sent), max_len)  # Truncate if longer than max_len
        sents[i, :length] = sent[:length]  # Padding and truncation

    # Convert to tensor and store original indices for restoration
    sents = torch.from_numpy(sents)
    return sents, torch.tensor(original_idx)