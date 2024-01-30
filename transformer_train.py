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
import gc


import matplotlib.pyplot as plt
import matplotlib


from audiolm_pytorch.data import get_audio_dataset
from experiment_config import ds_folders, ds_buffer



device = 'cuda'
from_scratch = True # train model from scratch, otherwise load from checkpoint
ds_from_scratch = False # create data set dump from scratch (set True if data set or pre processing has changed)

num_passes = 500 # num passes through the dataset

learning_rate = 9e-5 # max learning rate
weight_decay = 0.05
beta1 = 0.9
beta2 = 0.95
batch_size = 512

seed = 1234


# visualization ideas
# 2d only: export training points and classify each background-image pixel with nearest neighbor classifier for visualizing the class distribution on a html canvas element for the user to select a 2d point with the mouse
# 2,5d: extend the 2d idea with a slider next to the visualization for changing the z-direction. Changing the slider value could also change the calculated color distribution and the slider color could represent which classes are present at a specific z coordinate
# 4d: two 2d plots next to each other for selecting 4d point. Problem: User could select hihat in one and kick in the other plot. Weired results? Maybe a training of "mixed samples" is possible for compensate this?


# implement stop token for data class (variable length sequences)

# generate a sample map by choosing from the 2d embedding a real sample (by nearest neighbor) for checking if the 2d embedding actually has a meaningful representation of the samples
# distance between generated and related real sample
# embedding with 64 time shape, 1D convolutions in time dimension like in encodec paper
# vae loss that cares more about the first time steps (because they are more important for the sound)
# gan architecture
# Randomizing generation in transformer for more variety (not most probable sample is taken but randomly from best 5 or so)
# consider quantized output of encodec model (one torch embedding for each index dimension)
# decoder only?

# mehr samples anhoeren und auch label checken
# try out fixed randomness for training vae
# limit 2d embedding to fixed rectangle, train transformer regarding this rectangle

# style transfer, style gan paper nochmal lesen
# 2d embedding: 2d embedding with 64 time shape
# evaluation metrices:

# loss
# user listening tests (which sounds better?)
# generate audio for all the 2d points of the training data and compare with the original audio embedding. How accurate is it? For in between points, how similar is it to nearby points? We want to have a novelty factor in the generated audio, but it should still be similar to the original audio


torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)


config = dict(
    block_size = 150,
    block_size_condition = 2,
    vocab_size = 128,
    n_layer = 10,
    n_head = 5,
    n_embd = 240,
    dropout = 0.35,
    bias = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
)

def load_model(ckpt_path):
    global config
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['config']
    # force these config attributes to be equal otherwise we can't even resume training
    # the rest of the attributes (e.g. dropout) can stay as desired from command line
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'block_size_condition', 'bias', 'vocab_size']:
        config[k] = checkpoint_model_args[k]
    # create the model
    model = gpt.ConditionedGPT(config)
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


def save_model(checkpoint_path, model, optimizer, i, best_val_loss, config):
    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'iter_num': i+1,
        'best_val_loss': best_val_loss,
        'config': config,
    }
    # print(f"saving checkpoint to results/ckpt.pt with val loss {best_val_loss}")
    torch.save(checkpoint, checkpoint_path)



if __name__ == '__main__':

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

    # calculate loss of model for a given dataset (executed during training)
    @torch.no_grad()
    def det_loss_testing(ds, model, condition_model):
        dl = DataLoader(ds, batch_size=batch_size, shuffle=False) # get new dataloader because we want random sampling here!
        model.eval()
        losses = []
        for dx, dy, _, _ in dl:
            dx = dx.to(device)
            dy = dy.to(device)
            condition_bottleneck = condition_model(dx,True)[0]
            _, loss = model.forward(dx,dy, condition_bottleneck)
            losses.append(loss.cpu().detach().item())    
        model.train()
        return np.mean(losses)


    def train():
        # load model
        # training from scratch
        start_iter = 0
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
        
        condition_model = torch.load('results/vaes/vae_100.pt')
        condition_model.eval()

        # do not change
        best_val_loss = 1e9
        best_val_loss_iter = 0
        best_train_loss = 1e9
        best_train_loss_iter = 0

        train_losses = []
        val_losses = []
        change_learning_rate = learning_rate
        actual_learning_rate = learning_rate


        model.train()
        iteration = 0
        for i in range(start_iter,num_passes):
            iteration = i
            for dx, dy, _, _ in dl_train: # training is unsupervised so we don't need the labels (only shifted x)
                dx = dx.to(device)
                dy = dy.to(device)
                condition_bottleneck = condition_model(dx,True)[0]
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
            
            
            # change learning rate at several points during training
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
                # print('changed learning rate to %.3e at pass %d' % (change_learning_rate, i))
                optimizer = model.configure_optimizers(weight_decay=weight_decay,learning_rate=change_learning_rate,betas=(beta1, beta2),device_type=device)
                actual_learning_rate = change_learning_rate
            
            # plot training stats
            if i % 2 == 0:
                train_loss = det_loss_testing(ds_train, model, condition_model)
                val_loss = det_loss_testing(ds_val, model, condition_model)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                # print('ds pass: %d\ttrain loss: %.5f\tval loss: %.5f' % (i, train_loss, val_loss))

                if i > 0 and val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_val_loss_iter = i
                    save_model('results/ckpt.pt', model, optimizer, i, best_val_loss, config)
                if i > 0 and train_loss < best_train_loss:
                    best_train_loss = train_loss
                    best_train_loss_iter = i

                    
                    
                # save losses plot
                plt.close(0)
                plt.figure(0)
                plt.plot(train_losses, label='train')
                plt.plot(val_losses, label='val')
                plt.legend()
                plt.savefig('results/losses.png')
                # plt.show()
                
                # early stopping
                if i > best_val_loss_iter + 10:
                    print('early stopping at pass %d' % i)
                    break
                
        del model
        del condition_model
        gc.collect()

        return iteration, best_val_loss, best_val_loss_iter, best_train_loss, best_train_loss_iter
    
    # doing a random search
    with open('results/parameter_search.csv', 'w') as f:
        f.write("n_head,n_layer,n_embd,learning_rate,dropout,iteration,best_val_loss,best_val_loss_iter,best_train_loss,best_train_loss_iter\n")

    for ran_trial in range(2000):
        config['n_head'] = random.choice([1,2,3,4,5,6,8,10,12])
        config['n_layer'] = random.randint(1,12)
        
        config['n_embd'] = random.randint(10,50)*10
        while config['n_embd'] % config['n_head'] != 0:
            config['n_embd'] = random.randint(10,50)*10
        learning_rate = 10**random.uniform(-5,-3)
        config['dropout'] = random.uniform(0,0.5)
        
        # print out selected parameters
        print(f"n_head: {config['n_head']}, n_layer: {config['n_layer']}, n_embd: {config['n_embd']}, learning_rate: {learning_rate}, dropout: {config['dropout']}")

        iteration, best_val_loss, best_val_loss_iter, best_train_loss, best_train_loss_iter = -1, -1, -1, -1, -1
        try:
            iteration, best_val_loss, best_val_loss_iter, best_train_loss, best_train_loss_iter = train()
            print(ran_trial, best_val_loss)
        except:
            print("failed")
        
        # save all parameters and results in csv file:
        with open('results/parameter_search.csv', 'a') as f:
            f.write(f"{config['n_head']},{config['n_layer']},{config['n_embd']},{learning_rate},{config['dropout']},{iteration},{best_val_loss},{best_val_loss_iter},{best_train_loss},{best_train_loss_iter}\n")
        
            
            
        print("")
        torch.cuda.empty_cache()
    
    
    
    # doing a gridsearch
    # if n_head is 8, there are gpu memory limits. Why??
    
    # for n_head in [3,8,12]:
    #     for n_layer in [12,8,5]:
    #         config['n_head'] = n_head
    #         config['n_layer'] = n_layer
    #         try:
    #             best_val_loss = train()
    #             print(f"n_head: {n_head}, n_layer: {n_layer}, best_val_loss: {best_val_loss}")
    #         except:
    #             print(f"n_head: {n_head}, n_layer: {n_layer}, failed")
    #         torch.cuda.empty_cache()
