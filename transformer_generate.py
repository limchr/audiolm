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
import matplotlib.patches as mpatches

from experiment_config import ds_folders, ds_buffer, ckpt_vae, ckpt_transformer, ckpt_transformer_latest, ckpt_discriminator

import os
import shutil

outdir = 'results/samples'
ckpt = ckpt_transformer

device = 'cuda'
seed = 1234

from experiment_config import ns


num_generate = 150
num_sample_points_export = 5000

# visualization_area = [-0.8, 0.5, -0.6, 0.4] # area to be sampled (where training data is within the embedding space xmin, xmax, ymin, ymax)
visualization_area = [-0.9, 0.9, -0.9, 0.9]
visualization_to_model_space = lambda va,x,y: [ (va[1]-va[0]) * (x+1)/2 + va[0], (va[3]-va[2]) * (y+1)/2 + va[2] ]
model_to_visualization_space = lambda va,x,y: [ (x-va[0])/(va[1]-va[0]) * 2 - 1, (y-va[2])/(va[3]-va[2]) * 2 - 1 ]
sampling_x = np.linspace(visualization_area[0],visualization_area[1],ns)
sampling_y = np.linspace(visualization_area[2],visualization_area[3],ns)


torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)


if os.path.exists(outdir) and os.path.isdir(outdir):
    shutil.rmtree(outdir)
os.makedirs(os.path.join(outdir,'samples'))
os.makedirs(os.path.join(outdir,'nn'))
os.makedirs(os.path.join(outdir,'vae'))


from transformer_train_torch import GesamTransformer
model = torch.load(ckpt)[0]
model.eval()

condition_model = torch.load(ckpt_vae)
condition_model.eval()

discriminator = torch.load(ckpt_discriminator)
discriminator.eval()



# get the audio dataset
dsb, ds_train, ds_val, dl_train, dl_val = get_audio_dataset(audiofile_paths= ds_folders,
                                                                dump_path= ds_buffer,
                                                                build_dump_from_scratch=False,
                                                                only_labeled_samples=True,
                                                                test_size=0.1,
                                                                equalize_class_distribution=True,
                                                                equalize_train_data_loader_distribution=True,
                                                                batch_size=512,
                                                                seed=seed)
# check for train test split random correctness
print(ds_train[123][3], ds_train[245][3], ds_val[456][3], ds_val[125][3])



# 
# generate zy.csv for visualization of training embedding points in the web app
# 
bottlenecks = []
numeric_classes = []
for d in ds_train:
    dx = d[0].unsqueeze(0).to(device=device)
    condition_bottleneck = condition_model(dx,True)[0][0].cpu().detach().numpy()
    bottlenecks.append(condition_bottleneck)
    numeric_classes.append(d[3].item())

bottlenecks = np.array(bottlenecks, dtype=np.float32)
# project bottlenecks to visualization space with model to visualization space function
numeric_classes = np.array(numeric_classes, dtype=np.int32)


plt.scatter(bottlenecks[:,0], bottlenecks[:,1])
plt.xlim(-1,1)
plt.ylim(-1,1)
plt.savefig('results/visualization_train_embedding.png')


# export bottlenecks for visualization in web app
bottlenecks_visu = np.array([model_to_visualization_space(visualization_area, x[0], x[1]) for x in bottlenecks])
zy_combined = np.hstack((bottlenecks_visu,(numeric_classes)[None].T))
# sample 1000 random points from zy_combined to reduce the file size
zy_combined = zy_combined[np.random.choice(zy_combined.shape[0], num_sample_points_export, replace=False),:]
np.savetxt('results/zy.csv', zy_combined, delimiter=',', newline='\n', fmt='%.6f')


def classify(x):
    softmax = discriminator(x,True)[0]
    argsort = torch.argsort(softmax)
    margin = (softmax[argsort[-1]]-softmax[argsort[-2]]).item()
    best_confidence = softmax[argsort[-1]].item()
    class_id = argsort[-1].item()
    return class_id, margin


#
# generate samples in a grid
# 

generated_map = []
nneighbors = 5

for xi,x in enumerate(sampling_x):
    generated = []
    for yi,y in enumerate(sampling_y):
        gx = None
        gx = model.generate(num_generate=num_generate, condition=[[x,y]])

        vae_gx = condition_model.decode(torch.tensor([[x,y]],dtype=torch.float).to(device))

        sort_i = np.argsort(np.linalg.norm(bottlenecks - np.array([x,y],dtype=np.float32), axis=1))[:nneighbors]
        nn_gx = torch.zeros((1,150,128)).to(device)
        for i in sort_i:
            dxn = ds_train[i][0].to(device)
            nn_gx += dxn
        nn_gx /= nneighbors

        p_transformer = classify(gx)
        p_vae = classify(vae_gx)
        p_nn = classify(nn_gx)
        print(p_transformer)

        generated.append({'transformer': gx, 'vae': vae_gx, 'nn': nn_gx, 'classifications': {'transformer': p_transformer, 'vae': p_vae, 'nn': p_nn}})

        print('generated %d/%d' % (xi*ns+yi, ns*ns))
        if True:
            wav = dsb.dataset.dataset.decode_sample(gx.to('cpu'))
            dsb.dataset.dataset.save_audio(wav, f'results/samples/samples/generated_%05d_%05d.wav' % (xi,yi))
            
            wav = dsb.dataset.dataset.decode_sample(vae_gx.to('cpu'))
            dsb.dataset.dataset.save_audio(wav, f'results/samples/vae/generated_%05d_%05d.wav' % (xi,yi))

            wav = dsb.dataset.dataset.decode_sample(nn_gx.to('cpu'))
            dsb.dataset.dataset.save_audio(wav, f'results/samples/nn/generated_%05d_%05d.wav' % (xi,yi))


    generated_map.append(generated)



import pickle as pkl
with open('results/map_data.pkl', 'wb') as f:
    pkl.dump(generated_map, f)


# 
# instruction for exporting data to web app
#

print('For integrating the new data into the web app, do the following:')
print('copy zy.csv to the web app folder: cp /home/chris/src/audiolm/results/zy.csv /home/chris/src/audiogen_demo/data/models/drums/zy.csv')
print('remove the old sample files: rm /home/chris/src/audiogen_demo/data/models/drums/samples/ -r')
print('move new generated samples to the folder: mv /home/chris/src/audiolm/results/samples/samples /home/chris/src/audiogen_demo/data/models/drums/')
print('add classes to the web app:')
class_str = ''
for i in range(len(dsb.dataset.dataset.classes)):
    class_str += f'"{dsb.dataset.dataset.classes[i]}",'
print(class_str)
