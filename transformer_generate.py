import torch
import torch.nn.functional as F
import torchaudio


from audiolm_pytorch.data import get_audio_dataset
from torch.utils.data import DataLoader, random_split
from dataclasses import dataclass

import audiolm_pytorch.gpt as gpt

from audiolm_pytorch import EncodecWrapper

import random
import numpy as np

import matplotlib.pyplot as plt
import matplotlib

from transformer_train import load_model, AudioConfig

import os
import shutil

outdir = 'results/samples'

if os.path.exists(outdir) and os.path.isdir(outdir):
    shutil.rmtree(outdir)
os.makedirs(outdir)


device = 'cuda'
seed = 1234
ns = 50 # number of samples for x and y (total samples = ns*ns)
num_generate = 150

torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

model, checkpoint = load_model('results/ckpt.pt')
config = model.config

ss = np.linspace(-1,1,ns)

from experiment_config import ds_folders, ds_buffer

# get the audio dataset
dsb, ds_train, ds_val, dl_train, dl_val = get_audio_dataset(audiofile_paths= ds_folders,
                                                                dump_path= ds_buffer,
                                                                build_dump_from_scratch=False,
                                                                only_labeled_samples=True,
                                                                test_size=0.2,
                                                                equalize_class_distribution=True,
                                                                equalize_train_data_loader_distribution=True,
                                                                batch_size=512,
                                                                seed=seed)



condition_cnn = torch.load('results/cnn_model.pt')
condition_cnn.eval()


# generate zy.csv for visualization in the web app
bottlenecks = []
numeric_classes = []
for d in dsb:
    dx = d[0].unsqueeze(0)
    condition_bottleneck = condition_cnn(dx[:,:32,:],True)[0].cpu().detach().numpy()
    bottlenecks.append(condition_bottleneck)
    numeric_classes.append(d[3].item())

bottlenecks = np.array(bottlenecks, dtype=np.float32)
numeric_classes = np.array(numeric_classes, dtype=np.int32)

zy_combined = np.hstack((bottlenecks,(numeric_classes)[None].T))
np.savetxt('results/zy.csv', zy_combined, delimiter=',', newline='\n', fmt='%.6f')

# plt.scatter(bottlenecks[:,0], bottlenecks[:,1])
# plt.xlim(-1,1)
# plt.ylim(-1,1)
for xi,x in enumerate(ss):
    for yi,y in enumerate(ss):
        gx = torch.zeros((1,1,config.vocab_size), dtype=torch.float32).to(device)
    # gx[:,:,:] = dx[clsi%dx.shape[0],0,:]
        for i in range(num_generate):
            ng = model.forward(gx, None, torch.tensor([[x,y]], dtype=torch.float32).to(device=device))[0]
            gx = torch.cat((gx, ng), dim=1)
            
        print('generated %d/%d' % (xi*ns+yi, ns*ns))
        wav = dsb.dataset.dataset.decode_sample(gx)
        dsb.dataset.dataset.save_audio(wav, f'results/samples/generated_%05d_%05d.wav' % (xi,yi))


print('For integrating the new data into the web app, do the following:')
print('copy zy.csv to the web app folder: cp /home/chris/src/audiolm/results/zy.csv /home/chris/src/audiogen_demo/data/models/drums/zy.csv')
print('remove the old sample files: rm /home/chris/src/audiogen_demo/data/models/drums/samples/ -r')
print('move new generated samples to the folder: mv /home/chris/src/audiolm/results/samples /home/chris/src/audiogen_demo/data/models/drums/')






print('add classes to the web app:')
class_str = ''
for i in range(len(dsb.dataset.dataset.classes)):
    class_str += f'"{dsb.dataset.dataset.classes[i]}",'
print(class_str)

pass