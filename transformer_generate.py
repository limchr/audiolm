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

from transformer_train import load_model
from experiment_config import ds_folders, ds_buffer, ckpt_vae, ckpt_transformer

import os
import shutil

outdir = 'results/samples'

device = 'cuda'
seed = 1234
ns = 80 # number of samples for x and y (total samples = ns*ns)
num_generate = 150
num_sample_points_export = 2500

visualization_area = [-0.8, 0.5, -0.6, 0.4] # area to be sampled (where training data is within the embedding space xmin, xmax, ymin, ymax)

visualization_to_model_space = lambda va,x,y: [ (va[1]-va[0]) * (x+1)/2 + va[0], (va[3]-va[2]) * (y+1)/2 + va[2] ]
model_to_visualization_space = lambda va,x,y: [ (x-va[0])/(va[1]-va[0]) * 2 - 1, (y-va[2])/(va[3]-va[2]) * 2 - 1 ]
sampling_x = np.linspace(visualization_area[0],visualization_area[1],ns)
sampling_y = np.linspace(visualization_area[2],visualization_area[3],ns)


torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)


if os.path.exists(outdir) and os.path.isdir(outdir):
    shutil.rmtree(outdir)
os.makedirs(outdir)



model, checkpoint = load_model(ckpt_transformer)
config = model.config

condition_model = torch.load(ckpt_vae)
condition_model.eval()





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



# 
# generate zy.csv for visualization of training embedding points in the web app
# 
bottlenecks = []
numeric_classes = []
for d in dsb:
    dx = d[0].unsqueeze(0).to(device=device)
    condition_bottleneck = condition_model(dx,True)[0][0].cpu().detach().numpy()
    bottlenecks.append(condition_bottleneck)
    numeric_classes.append(d[3].item())

bottlenecks = np.array(bottlenecks, dtype=np.float32)
# project bottlenecks to visualization space with model to visualization space function
bottlenecks = np.array([model_to_visualization_space(visualization_area, x[0], x[1]) for x in bottlenecks])
numeric_classes = np.array(numeric_classes, dtype=np.int32)
zy_combined = np.hstack((bottlenecks,(numeric_classes)[None].T))
# sample 1000 random points from zy_combined
zy_combined = zy_combined[np.random.choice(zy_combined.shape[0], num_sample_points_export, replace=False),:]
np.savetxt('results/zy.csv', zy_combined, delimiter=',', newline='\n', fmt='%.6f')

plt.scatter(bottlenecks[:,0], bottlenecks[:,1])
plt.xlim(-1,1)
plt.ylim(-1,1)
plt.savefig('results/visualization_train_embedding.png')

#
# generate samples in a grid
# 

for xi,x in enumerate(sampling_x):
    for yi,y in enumerate(sampling_y):
        gx = torch.zeros((1,1,config.vocab_size), dtype=torch.float32).to(device)
    # gx[:,:,:] = dx[clsi%dx.shape[0],0,:]
        for i in range(num_generate):
            ng = model.forward(gx, None, torch.tensor([[x,y]], dtype=torch.float32).to(device=device))[0]
            gx = torch.cat((gx, ng), dim=1)
        print('generated %d/%d' % (xi*ns+yi, ns*ns))
        wav = dsb.dataset.dataset.decode_sample(gx.to('cpu'))
        dsb.dataset.dataset.save_audio(wav, f'results/samples/generated_%05d_%05d.wav' % (xi,yi))

# 
# instruction for exporting data to web app
#

print('For integrating the new data into the web app, do the following:')
print('copy zy.csv to the web app folder: cp /home/chris/src/audiolm/results/zy.csv /home/chris/src/audiogen_demo/data/models/drums/zy.csv')
print('remove the old sample files: rm /home/chris/src/audiogen_demo/data/models/drums/samples/ -r')
print('move new generated samples to the folder: mv /home/chris/src/audiolm/results/samples /home/chris/src/audiogen_demo/data/models/drums/')

print('add classes to the web app:')
class_str = ''
for i in range(len(dsb.dataset.dataset.classes)):
    class_str += f'"{dsb.dataset.dataset.classes[i]}",'
print(class_str)
