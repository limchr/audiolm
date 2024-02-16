import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
import gc
import matplotlib.pyplot as plt
from audiolm_pytorch.data import get_audio_dataset
from experiment_config import ds_folders, ds_buffer, ckpt_vae, ckpt_transformer, ckpt_transformer_latest
from torch.utils.data import DataLoader

seed = 1234
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)


device = 'cuda'
from_scratch = True # train model from scratch, otherwise load from checkpoint
ds_from_scratch = False # create data set dump from scratch (set True if data set or pre processing has changed)
only_labeled_samples=True

num_passes = 300 # num passes through the dataset

learning_rate = 9e-5 # max learning rate
weight_decay = 0.05
beta1 = 0.9
beta2 = 0.95
batch_size = 256


stats_every_iteration = 10
train_set_testing_size = 1000
is_gan_training = False


config = dict(
    block_size = 150,
    block_size_encoder = 2,
    input_dimension = 128,
    internal_dimension = 512,
    feedforward_dimension = 2048,
    n_layer_encoder = 4,
    n_layer_decoder = 11,
    n_head = 8,
    dropout = 0.25
)

class GesamTransformer(nn.Module):
    def __init__(self, 
                 config,
                 device
                 ):
        super().__init__()
        self.config = config
        self.device = device
        
        self.input_projection_decoder = nn.Linear(config['input_dimension'], config['internal_dimension']).to(device)
        self.input_projection_encoder = nn.Linear(1,config['internal_dimension']).to(device)

        self.output_projection = nn.Linear(config['internal_dimension'], config['input_dimension']).to(device)

        # positional encoding layer decoder
        self.input_posemb_decoder = nn.Embedding(config['block_size'], config['internal_dimension']).to(device)
        # positional enocding layer encoder
        self.input_posemb_encoder = nn.Embedding(config['block_size_encoder'], config['internal_dimension']).to(device)
        
        self.transformer = nn.Transformer(
                d_model = config['internal_dimension'], 
                batch_first=True, 
                nhead=config['n_head'],
                num_encoder_layers=config['n_layer_encoder'],
                num_decoder_layers=config['n_layer_decoder'],
                dropout=config['dropout'],
                dim_feedforward=config['feedforward_dimension']
        ).to(device)


    def forward(self, xdec, xenc):
        
        xdec = self.input_projection_decoder(xdec)
        pos = torch.arange(0, xdec.shape[1], dtype=torch.long).to(self.device)
        pos_emb_dec = self.input_posemb_decoder(pos)
        xdec = xdec + pos_emb_dec
        
        xenc = self.input_projection_encoder(xenc.unsqueeze(-1))
        pos = torch.arange(0, self.config['block_size_encoder'], dtype=torch.long).to(self.device)
        pos_emb_enc = self.input_posemb_encoder(pos)
        xenc = xenc + pos_emb_enc
        
        mask = self.get_tgt_mask(xdec.shape[1])
        ydec = self.transformer.forward(src=xenc,tgt=xdec,tgt_mask=mask)
        ydec = self.output_projection(ydec)
        
        return ydec
    
    def predict(self, cmodel, dx, dy):
            dx = dx.to(device).detach()
            dy = dy.to(device).detach()
            cx = cmodel(dx,True)[0].detach()
            logits = self.forward(xdec=dx,xenc=cx)
            loss = F.mse_loss(logits,dy)
            return logits, loss

    @torch.no_grad()
    def generate(self, num_generate, condition):
        """
        num_generate: output size of generated sequence (time dimension)
        condition: nx2 list of tensor (n=batch size)
        """
        if not torch.is_tensor(condition):
            condition = torch.tensor(condition, dtype=torch.float32).to(device=self.device)        
        gx = torch.zeros((condition.shape[0],1,self.config['input_dimension']), dtype=torch.float32).to(device) # all zeros is the 'start token'
            
        for _ in range(num_generate-1):
            ng = self.forward(xdec=gx,xenc=condition)[:, [-1], :]
            gx = torch.cat((gx, ng), dim=1)
        return gx
    

    def get_tgt_mask(self, size) -> torch.tensor:
        mask = torch.tril(torch.ones(size, size) == 1) # Lower triangular matrix
        mask = mask.float()
        mask = mask.masked_fill(mask == 0, float('-inf')) # Convert zeros to -inf
        mask = mask.masked_fill(mask == 1, float(0.0)) # Convert ones to 0
        
        return mask



if __name__ == '__main__':

    # # for debugging
    # ds_folders = ['/home/chris/data/audio_samples/ds_min/']
    # ds_buffer = '/home/chris/data/audio_samples/dsmin.pkl'

    # get the audio dataset
    dsb, ds_train, ds_val, dl_train, dl_val = get_audio_dataset(audiofile_paths= ds_folders,
                                                                    dump_path= ds_buffer,
                                                                    build_dump_from_scratch=ds_from_scratch,
                                                                    only_labeled_samples=only_labeled_samples,
                                                                    test_size=0.1,
                                                                    equalize_class_distribution=True,
                                                                    equalize_train_data_loader_distribution=True,
                                                                    batch_size=batch_size,
                                                                    seed=seed)
    # check for train test split random correctness
    print(ds_train[12][3], ds_train[23][3], ds_val[1][3], ds_val[2][3])


    # calculate loss of model for a given dataset (executed during training)
    @torch.no_grad()
    def det_loss_testing(ds, model, condition_model):
        dl = DataLoader(ds, batch_size=batch_size, shuffle=False) # get new dataloader because we want random sampling here!
        losses = []
        gen_losses = []
        gen_loss_crop = 150
        for dx, dy, _, _ in dl:
            dx = dx.to(device)
            dy = dy.to(device)
            logits, loss = model.predict(condition_model, dx, dy)
            losses.append(loss.cpu().detach().item())    
            condition_bottleneck = condition_model(dx,True)[0].detach()
            gx = model.generate(gen_loss_crop, condition_bottleneck)
            gen_loss = torch.mean(torch.abs(dx[:,:gen_loss_crop,:]-gx[:,:gen_loss_crop,:])).item()
            gen_losses.append(gen_loss)
        return np.mean(losses), np.mean(gen_losses)

    def save_model( ckpt_path,
                    model,
                    iter,
                    config,
                    iterations,
                    train_losses,
                    val_losses,
                    train_gen_losses,
                    val_gen_losses,
                    best_val_loss,
                    best_val_loss_iter,
                    best_train_loss,
                    best_train_loss_iter,
                    best_train_gen_loss,
                    best_train_gen_loss_iter,
                    best_val_gen_loss,
                    best_val_gen_loss_iter):
        dump = [    model,
                    iter+1,
                    config,
                    iterations,
                    train_losses,
                    val_losses,
                    train_gen_losses,
                    val_gen_losses,
                    best_val_loss,
                    best_val_loss_iter,
                    best_train_loss,
                    best_train_loss_iter,
                    best_train_gen_loss,
                    best_train_gen_loss_iter,
                    best_val_gen_loss,
                    best_val_gen_loss_iter
                    ]
        torch.save(dump, ckpt_path)
        


    def load_model(ckpt_path):
        return torch.load(ckpt_path)



    def train(is_parameter_search):
        global config

        # load model
        # training from scratch
        
        # do not change
        best_val_loss = 1e9
        best_val_loss_iter = 0
        best_train_loss = 1e9
        best_train_loss_iter = 0
        best_train_gen_loss = 1e9
        best_train_gen_loss_iter = 0
        best_val_gen_loss = 1e9
        best_val_gen_loss_iter = 0

        iterations = [] # for plotting
        train_losses = []
        val_losses = []
        train_gen_losses = []
        val_gen_losses = []
        change_learning_rate = learning_rate
        actual_learning_rate = learning_rate


        
        start_iter = 0
        if from_scratch:
            model = GesamTransformer(config=config, device=device)
            model.to(device)
        # training from checkpoint
        else: 
            print(f"Resuming training from checkpoint")
            model,\
            start_iter,\
            config,\
            iterations,\
            train_losses,\
            val_losses,\
            train_gen_losses,\
            val_gen_losses,\
            best_val_loss,\
            best_val_loss_iter,\
            best_train_loss,\
            best_train_loss_iter,\
            best_train_gen_loss,\
            best_train_gen_loss_iter,\
            best_val_gen_loss,\
            best_val_gen_loss_iter = load_model(ckpt_transformer_latest)
            # model = torch.load(ckpt_transformer)

        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=(beta1,beta2))
        
        condition_model = torch.load(ckpt_vae).to(device)
        condition_model.eval()


        
        model.train()
        iteration = 0
        for i in range(start_iter,num_passes):
            iteration = i
            model.train()
            print('start iteration %d' % i)
            for dx, dy, _, _ in dl_train: # training is unsupervised so we don't need the labels (only shifted x)
                # autoregressive loss transformer training
                optimizer.zero_grad()
                logits, loss = model.predict(condition_model, dx, dy)                
                loss.backward()
                optimizer.step()
                
            print('iteration %d done' % i)

                    
            # change learning rate at several points during training
            if i > int(num_passes*0.9):
                change_learning_rate = learning_rate * 0.5 * 0.5 * 0.5 * 0.5 * 0.5
            elif i > int(num_passes*0.8):
                change_learning_rate = learning_rate * 0.5 * 0.5 * 0.5 * 0.5
            elif i > int(num_passes*0.7):
                change_learning_rate = learning_rate * 0.5 * 0.5 * 0.5
            elif i > int(num_passes*0.6):
                change_learning_rate = learning_rate * 0.5 * 0.5
            elif i > int(num_passes*0.5):
                change_learning_rate = learning_rate * 0.5
            if change_learning_rate != actual_learning_rate:   
                optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=(beta1,beta2))

                actual_learning_rate = change_learning_rate
                if not is_parameter_search:
                    print('changed learning rate to %.3e at pass %d' % (change_learning_rate, i))


            # plot training stats
            if i > 0 and (i+1) % stats_every_iteration == 0:

                model.eval()

                print('calculating losses at iteration %d' % i)
                train_loss, train_gen_loss = det_loss_testing(
                    torch.utils.data.Subset(ds_train,list(range(0,min(len(ds_train), train_set_testing_size)))), model, condition_model)
                val_loss, val_gen_loss = det_loss_testing(ds_val, model, condition_model)
                
                iterations.append(i) # for plotting
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                train_gen_losses.append(train_gen_loss)
                val_gen_losses.append(val_gen_loss)
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_val_loss_iter = i
                    # if not is_parameter_search:
                    #     print('saving model to %s with val loss %.5f' % (ckpt_transformer, best_val_loss))
                    #     save_model(ckpt_transformer, model, optimizer, i, best_val_loss, config)
                if train_loss < best_train_loss:
                    best_train_loss = train_loss
                    best_train_loss_iter = i
                if val_gen_loss < best_val_gen_loss:
                    best_val_gen_loss = val_gen_loss
                    best_val_gen_loss_iter = i
                    if not is_parameter_search:
                        print('saving model to %s with val MAE loss %.5f' % (ckpt_transformer, val_gen_loss))
                        save_model( ckpt_transformer, model, i, config, iterations, train_losses, val_losses, train_gen_losses, val_gen_losses,\
                                    best_val_loss, best_val_loss_iter, best_train_loss, best_train_loss_iter, best_train_gen_loss, best_train_gen_loss_iter, best_val_gen_loss, best_val_gen_loss_iter
                                    )

                if train_gen_loss < best_train_gen_loss:
                    best_train_gen_loss = train_gen_loss
                    best_train_gen_loss_iter = i

                if not is_parameter_search:
                    print("##### iteration %d/%d" % (i, num_passes))
                    print('train loss: %.5f (best train loss: %.5f at iter %i)' % (train_loss, best_train_loss, best_train_loss_iter))
                    print('train MAE loss: %.5f (best train MAE loss: %.5f at iter %i)' % (train_gen_loss, best_train_gen_loss, best_train_gen_loss_iter))
                    print('test loss: %.5f (best test loss: %.5f at iter %i)' % (val_loss, best_val_loss, best_val_loss_iter))
                    print('test MAE loss: %.5f (best test MAE loss: %.5f at iter %i)' % (val_gen_loss, best_val_gen_loss, best_val_gen_loss_iter))

                    print('saving model to %s with val MAE loss %.5f' % (ckpt_transformer_latest, val_gen_loss))
                    save_model( ckpt_transformer_latest, model, i, config, iterations, train_losses, val_losses, train_gen_losses, val_gen_losses,\
                                best_val_loss, best_val_loss_iter, best_train_loss, best_train_loss_iter, best_train_gen_loss, best_train_gen_loss_iter, best_val_gen_loss, best_val_gen_loss_iter
                                )



                    print('')
                    
                    # save losses plot
                    plt.close(0)
                    plt.figure(0)
                    plt.subplots(2, 1, figsize=(10, 20))
                    plt.subplot(2,1,1)
                    plt.plot(iterations, train_losses, marker='o', linestyle='--', label='train')
                    plt.plot(iterations, val_losses, marker='o', linestyle='--', label='val')
                    plt.ylim(0,0.5)
                    plt.legend()
                    plt.title('Train and Validation Losses after epoch %d' % i)

                    # Create the second subplot for train_gen_losses and val_gen_losses
                    plt.subplot(2, 1, 2)
                    plt.plot(iterations, train_gen_losses, marker='o', linestyle='--', label='train MAE')
                    plt.plot(iterations, val_gen_losses, marker='o', linestyle='--', label='val MAE')
                    plt.ylim(0,0.5)
                    plt.legend()
                    plt.title('Train and Validation MAE Losses after epoch %d' % i)

                    plt.tight_layout()  # Adjust layout to prevent overlapping
                    plt.savefig('results/losses.png')  # Save the figure
                    # plt.show()
                                        
                # early stopping
                if is_parameter_search and i > best_val_loss_iter + 30:
                    print('early stopping at pass %d' % i)
                    break


                
        del model
        del condition_model
        gc.collect()

        return iteration, best_val_loss, best_val_loss_iter, best_train_loss, best_train_loss_iter
    
    
    # doing a random search
    
    def random_parameter_search():
        random.seed(None) # reset seed to current time
        
        # # initialize new csv file
        # with open('results/parameter_search.csv', 'w') as f:
        #     f.write("n_head,n_layer,n_embd,learning_rate,dropout,iteration,best_val_loss,best_val_loss_iter,best_train_loss,best_train_loss_iter\n")

        for ran_trial in range(2000):
            config['n_head'] = random.choice([5,6,8,10,12,14])
            config['n_layer'] = random.randint(14,22)
            
            config['n_embd'] = random.randint(30,50)*10
            while config['n_embd'] % config['n_head'] != 0:
                config['n_embd'] = random.randint(30,50)*10
            learning_rate = random.uniform(0.0001,0.00001)
            config['dropout'] = random.uniform(0.1,0.25)
            
            # print out selected parameters
            print(f"n_head: {config['n_head']}, n_layer: {config['n_layer']}, n_embd: {config['n_embd']}, learning_rate: {learning_rate}, dropout: {config['dropout']}")

            iteration, best_val_loss, best_val_loss_iter, best_train_loss, best_train_loss_iter = -1, -1, -1, -1, -1
            try:
                iteration, best_val_loss, best_val_loss_iter, best_train_loss, best_train_loss_iter = train(is_parameter_search=True)
                print(ran_trial, best_val_loss)
            except:
                print("failed")
            
            # save all parameters and results in csv file:
            with open('results/parameter_search.csv', 'a') as f:
                f.write(f"{config['n_head']},{config['n_layer']},{config['n_embd']},{learning_rate},{config['dropout']},{iteration},{best_val_loss},{best_val_loss_iter},{best_train_loss},{best_train_loss_iter}\n")
            
                
            print("")
            torch.cuda.empty_cache()
    
    train(is_parameter_search=False)
    # random_parameter_search()