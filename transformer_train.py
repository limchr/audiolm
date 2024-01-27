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

device = 'cuda'
from_scratch = True # train model from scratch, otherwise load from checkpoint
ds_from_scratch = False # create data set dump from scratch (set True if data set or pre processing has changed)

num_passes = 1200 # num passes through the dataset

start_iter = 0
learning_rate = 9e-5 # max learning rate
weight_decay = 0.05
beta1 = 0.9
beta2 = 0.95
batch_size = 512

seed = 1234


# do not change
best_val_loss = 1e9

# visualization ideas
# 2d only: export training points and classify each background-image pixel with nearest neighbor classifier for visualizing the class distribution on a html canvas element for the user to select a 2d point with the mouse
# 2,5d: extend the 2d idea with a slider next to the visualization for changing the z-direction. Changing the slider value could also change the calculated color distribution and the slider color could represent which classes are present at a specific z coordinate
# 4d: two 2d plots next to each other for selecting 4d point. Problem: User could select hihat in one and kick in the other plot. Weired results? Maybe a training of "mixed samples" is possible for compensate this?


# generate a sample map by choosing from the 2d embedding a real sample (by nearest neighbor) for checking if the 2d embedding actually has a meaningful representation of the samples
# distance between generated and related real sample
# embedding with 64 time shape, 1D convolutions in time dimension like in encodec paper
# try out VAE again with 1d convolutions
# gan architecture
# Randomizing generation in transformer for more variety (not most probable sample is taken but randomly from best 5 or so)
# consider quantized output of encodec model (one torch embedding for each index dimension)
# decoder only?

# mehr samples anhoeren und auch label checken
# va nochmal ausprobieren


# style transfer, style gan paper nochmal lesen
# 2d embedding: 2d embedding with 64 time shape
# evaluation metrices:

# loss
# user listening tests (which sounds better?)
# generate audio for all the 2d points of the training data and compare with the original audio embedding. How accurate is it? For in between points, how similar is it to nearby points? We want to have a novelty factor in the generated audio, but it should still be similar to the original audio


torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

@dataclass
class AudioConfig:
    block_size: int = 150
    block_size_condition: int = 2
    vocab_size: int = 128
    n_layer: int = 6
    n_head: int = 8
    n_embd: int = 240
    dropout: float = 0.35
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
config = AudioConfig()
model_args = dict(n_layer=config.n_layer, n_head=config.n_head, n_embd=config.n_embd, block_size=config.block_size, bias=config.bias, vocab_size=config.vocab_size, dropout=config.dropout) # start with model_args from command line


def load_model(ckpt_path):
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    # force these config attributes to be equal otherwise we can't even resume training
    # the rest of the attributes (e.g. dropout) can stay as desired from command line
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = checkpoint_model_args[k]
    # create the model
    gptconf = AudioConfig(**model_args)
    model = gpt.ConditionedGPT(gptconf)
    state_dict = checkpoint['model']
    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    model.to(device)
    return model, checkpoint


def save_model(checkpoint_path, model,optimizer,model_args,i,best_val_loss,config,sound_classes):
    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'model_args': model_args,
        'iter_num': i+1,
        'best_val_loss': best_val_loss,
        'config': config,
        'conditions': sound_classes
    }
    print(f"saving checkpoint to results/ckpt.pt with val loss {best_val_loss}")
    torch.save(checkpoint, checkpoint_path)


from audiolm_pytorch.data import get_audio_dataset

if __name__ == '__main__':

    from experiment_config import ds_folders, ds_buffer

    # get the audio dataset
    dsb, ds_train, ds_val, dl_train, dl_val = get_audio_dataset(audiofile_paths= ds_folders,
                                                                    dump_path= ds_buffer,
                                                                    build_dump_from_scratch=ds_from_scratch,
                                                                    only_labeled_samples=True,
                                                                    test_size=0.2,
                                                                    equalize_class_distribution=True,
                                                                    equalize_train_data_loader_distribution=True,
                                                                    batch_size=batch_size,
                                                                    seed=seed)

    # training from scratch
    if from_scratch:
        model = gpt.ConditionedGPT(config)
        model.to(device)
    # training from checkpoint
    else: 
        print(f"Resuming training from checkpoint")
        # resume training from a checkpoint.
        ckpt_path = 'results/ckpt.pt'
        model, checkpoint = load_model(ckpt_path)
        start_iter = checkpoint['iter_num']
        best_val_loss = checkpoint['best_val_loss']


    optimizer = model.configure_optimizers(weight_decay=weight_decay,learning_rate=learning_rate,betas=(beta1, beta2),device_type=device)

    condition_network = torch.load('results/vaes/vae_100.pt')
    condition_network.eval()

    @torch.no_grad()
    def det_loss_testing(ds, model):
        dl = DataLoader(ds, batch_size=batch_size, shuffle=False) # get new dataloader because we want random sampling here!

        model.eval()
        losses = []
        for dx, dy, _, _ in dl:
            condition_bottleneck = condition_network(dx,True)[0]
            _, loss = model.forward(dx,dy, condition_bottleneck)
            losses.append(loss.cpu().detach().item())    
        model.train()
        return np.mean(losses)

    train_losses = []
    val_losses = []
    change_learning_rate = learning_rate
    actual_learning_rate = learning_rate


    model.train()
    for i in range(start_iter,num_passes):
        for dx, dy, _, numeric_y in dl_train: # training is unsupervised so we don't need the labels (only shifted x)
            condition_bottleneck = condition_network(dx,True)[0]
            # rnds = torch.randn_like(condition_bottleneck).to(device) * 0.1
            # condition_z = condition_bottleneck+rnds
            condition_z = condition_bottleneck

            # # debug plot for condition_z:
            # cplt = condition_z.cpu().detach().numpy()
            # plt.scatter(cplt[:,0],cplt[:,1], c=numeric_y.cpu().detach().numpy(), cmap='tab10')
            # plt.savefig('results/condition_z.png')
            
            logits, loss = model.forward(dx,dy, condition_z)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            # if i % 100 == 0: print(loss.item()) # print more losses for debugging
        
        # if i%2 == 0 and i != 0:
        #     learning_rate = learning_rate * 5
        #     print('changed learning rate to %.3e at pass %d' % (learning_rate, i))
        #     optimizer = model.configure_optimizers(weight_decay=weight_decay,learning_rate=learning_rate,betas=(beta1, beta2),device_type=device)
        
        if i > int(num_passes*0.9):
            change_learning_rate = learning_rate * 0.6 * 0.6 * 0.6 * 0.6 * 0.6
        elif i > int(num_passes*0.8):
            change_learning_rate = learning_rate * 0.6 * 0.6 * 0.6 * 0.6
        elif i > int(num_passes*0.7):
            change_learning_rate = learning_rate * 0.6 * 0.6 * 0.6
        elif i > int(num_passes*0.6):
            change_learning_rate = learning_rate * 0.6 * 0.6
        elif i > int(num_passes*0.5):
            change_learning_rate = learning_rate * 0.6
        if change_learning_rate != actual_learning_rate:   
            print('changed learning rate to %.3e at pass %d' % (change_learning_rate, i))
            optimizer = model.configure_optimizers(weight_decay=weight_decay,learning_rate=change_learning_rate,betas=(beta1, beta2),device_type=device)
            actual_learning_rate = change_learning_rate
        
        if i % 2 == 0:
            train_loss = det_loss_testing(ds_train, model)
            val_loss = det_loss_testing(ds_val, model)
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            print('ds pass: %d\ttrain loss: %.5f\tval loss: %.5f' % (i, train_loss, val_loss))

            if i > 0 and val_loss < best_val_loss:
                best_val_loss = val_loss
                save_model('results/ckpt.pt', model,optimizer,model_args,i,best_val_loss,config,dsb.dataset.dataset.classes)
                
            # save losses plot
            plt.close(0)
            plt.figure(0)
            plt.plot(train_losses, label='train')
            plt.plot(val_losses, label='val')
            plt.legend()
            plt.savefig('results/losses.png')
            # plt.show()

