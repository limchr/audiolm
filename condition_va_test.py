
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.nn import functional as F

import math


import audiolm_pytorch.data as data
from dataclasses import dataclass
from audiolm_pytorch.condition_va import ConditionVA

import random
import numpy as np

import matplotlib.pyplot as plt
import matplotlib


device = 'cuda'
seed = 1234
batch_size = 32
num_passes = 200

torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)




va = ConditionVA().to(device)

ds = data.EncodecSoundDataset(folder='/home/chris/data/audio_samples/ds_extracted', seed=seed)
dsb = data.BufferedDataset(ds, '/home/chris/data/buffered_ds_extracted.pkl', False)
train_size = math.floor(0.9 * len(dsb))
val_size = math.floor(0.1 * len(dsb))
test_size = len(dsb) - train_size - val_size

ds_train, ds_test, ds_val = random_split(dsb, [train_size, test_size, val_size])
dl = DataLoader(ds_train, batch_size=batch_size, shuffle=True)
dl_val = DataLoader(ds_val, batch_size=batch_size, shuffle=True)





optimizer = torch.optim.Adam(va.parameters(), lr=3e-4)
loss_fn = nn.MSELoss(reduction='sum')


def det_loss(va,dl):
    losses = []
    va.eval()
    for d in dl:
        va.eval()
        x = d[0][:, :128, :] # crop sample in time dimension for easier calculation
        y = va.forward(x)
        loss = loss_fn(x, y)
        losses.append(loss.item())
    va.train()
    return losses


va.train()
for i in range(num_passes):
    for d in dl:
        x = d[0][:, :128, :] # crop sample in time dimension for easier calculation
        y = va.forward(x)
        loss = loss_fn(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        # print(loss.item())
    
    train_loss = np.mean(det_loss(va,dl))
    val_loss = np.mean(det_loss(va,dl_val))

    print('pass: %d \t train loss: %.4f \t val loss: %.4f' % (i,train_loss, val_loss))
        

print(va.encoder.fc4.weight)