import torch
import torch.nn.functional as F

import audiolm_pytorch.data as data
import experiment_config as ec

import numpy as np
import os

import matplotlib.pyplot as plt
import matplotlib

import random

seed = 1234

torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)


ds = data.EncodecSoundDataset(folders=ec.ds_folders, length=2, seed=seed, device='cpu')
dsb = data.BufferedDataset(ds, ec.ds_buffer, True)


print('faulty files:')
for f in ds.log_faulty_files: 
    print(f)
    os.rename(f, f + '_faulty')

print('no label: %i files (%.2f%%)' % (len(ds.log_no_label),100 * len(ds.log_no_label) / len(ds)))

for f in ds.log_no_label:
    print(f)
    

print('dublicate labels: %i files (%.2f%%)' % (len(ds.log_duplicate_labels),100 *  len(ds.log_duplicate_labels) / len(ds)))

for f in ds.log_duplicate_labels:
    print(f)



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
    
