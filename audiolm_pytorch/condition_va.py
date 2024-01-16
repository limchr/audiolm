import torch
import torch.nn as nn


import random
import numpy as np


device = 'cuda'
seed = 1234

torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

# vgl https://github.com/Jackson-Kang/Pytorch-VAE-tutorial/blob/master/01_Variational_AutoEncoder.ipynb

class ConditionEncoder(nn.Module):
    def __init__(self, hidden_dim, latent_dim, layers):
        super(ConditionEncoder, self).__init__()
        # Fully connected layers
        
        layers = [3840] + layers + [hidden_dim]
        self.layers = layers
        for i in range(len(layers)-1):
            setattr(self, 'fc{}'.format(i+1), nn.Linear(layers[i], layers[i+1]))
            setattr(self, 'bn{}'.format(i+1), nn.LayerNorm(layers[i+1]))
        
        
        self.mean = nn.Linear(hidden_dim, latent_dim)
        self.var = nn.Linear(hidden_dim, latent_dim)

        self.relu = torch.nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(0.2)


    def forward(self, x):
        x = x.view([x.shape[0], -1] )
        
        for i in range(1,len(self.layers)):
            x = getattr(self, 'fc{}'.format(i))(x)
            x = getattr(self, 'bn{}'.format(i))(x)
            x = self.relu(x)
            x = self.dropout(x)
        

        mean = self.mean(x)
        var = self.var(x)
       
        return mean, var

class ConditionDecoder(nn.Module):
    def __init__(self, hidden_dim, latent_dim, layers):
        super(ConditionDecoder, self).__init__()
        
        layers = [latent_dim, hidden_dim] + layers[::-1] + [3840]
        self.layers = layers
        
        # Fully connected layers
        for i in range(len(layers)-1):
            setattr(self, 'fc{}'.format(i+1), nn.Linear(layers[i], layers[i+1]))
            setattr(self, 'bn{}'.format(i+1), nn.LayerNorm(layers[i+1]))

        self.relu = torch.nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):

        for i in range(1,len(self.layers)-1):
            x = getattr(self, 'fc{}'.format(i))(x)
            x = getattr(self, 'bn{}'.format(i))(x)
            x = self.relu(x)
            x = self.dropout(x)

        x = getattr(self, 'fc{}'.format(len(self.layers)-1))(x)
        x = x.view([x.shape[0], 30, 128])

        return x

class ConditionVA(nn.Module):
    def __init__(self, hidden_dim, latent_dim, layers):
        super(ConditionVA, self).__init__()

        

        self.encoder = ConditionEncoder(hidden_dim, latent_dim, layers)
        self.decoder = ConditionDecoder(hidden_dim, latent_dim, layers)
        self.initialize_weights()
        

    def forward(self, x):

        mean, var = self.encoder(x)

        epsilon = torch.randn_like(var).to(device)        # sampling epsilon        
        z = mean + torch.exp(var / 2) * epsilon                         # reparameterization trick

        x_hat = self.decoder(z)
        return x_hat, mean, var


    def initialize_weights(self):
        for layer in list(self.encoder.children()) + list(self.decoder.children()):
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight, gain=0.2)
                nn.init.constant_(layer.bias, 0)  # Initialize biases to zero or another suitable value
