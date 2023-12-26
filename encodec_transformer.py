import torch
import torch.nn.functional as F
import torchaudio

import audiolm_pytorch.data as data
from torch.utils.data import DataLoader

import audiolm_pytorch.gpt as gpt

from audiolm_pytorch import EncodecWrapper



device = 'cuda'

num_passes = 20 # num passes through the dataset

learning_rate = 6e-4 # max learning rate
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95





# ds = data.EncodecSoundDataset(folder='/home/chris/data/audio_samples/ds_min')

ds = data.EncodecSoundDataset(folder='/home/chris/data/audio_samples/ds_extracted')
dl = DataLoader(ds, batch_size=4, shuffle=True)

config = gpt.AudioConfig()
model = gpt.GPT(config)
model.to(device)

optimizer = model.configure_optimizers(weight_decay=weight_decay,learning_rate=learning_rate,betas=(beta1, beta2),device_type=device)





for i in range(num_passes):
    for dx, dy, sample_class in dl:
        logits, loss = model.forward(dx,dy)
        loss.backward()
        print(loss)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

gx = torch.zeros((1,1,128)).to(device)
gx[:,:,:] = dx[0,0,:]

num_generate = 148

for i in range(num_generate):
    ng = model.forward(gx)[0]
    gx = torch.cat((gx, ng), dim=1)


encodec = EncodecWrapper()
encodec.to(device=device)
wav = encodec.decode(gx)
torchaudio.save('generated.wav', wav.to('cpu').reshape(1,-1), 24000)

        
pass