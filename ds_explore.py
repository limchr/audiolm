import torch
import torch.nn.functional as F
import torchaudio

import audiolm_pytorch.data as data
from torch.utils.data import DataLoader, random_split
from dataclasses import dataclass

import audiolm_pytorch.gpt as gpt

from audiolm_pytorch import EncodecWrapper

import random
import numpy as np

import matplotlib.pyplot as plt
import matplotlib

seed = 1234

ds = data.EncodecSoundDataset(folder='/home/chris/data/audio_samples/ds_extracted', seed=seed)
dsb = data.BufferedDataset(ds, '/home/chris/data/buffered_ds_extracted.pkl', True)

x = []
y = []

for d in dsb:
    x.append(d[0])
    y.append(d[2])

x = torch.stack(x)    
y = torch.stack(y)

ysum = y.sum(dim=0)

for ys,cn in zip(ysum, ds.class_groups.keys()):
    print('%s: %d' % (cn,ys.item()))
    
