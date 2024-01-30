import torch
import torch.nn as nn


import random
import numpy as np


device = 'cuda'
seed = 1234

# vgl https://github.com/Jackson-Kang/Pytorch-VAE-tutorial/blob/master/01_Variational_AutoEncoder.ipynb

class ConditionEncoder(nn.Module):
    def __init__(self, layers):
        super(ConditionEncoder, self).__init__()
        # Fully connected layers
        
        self.layers = layers
        for i in range(len(layers)-2):
            setattr(self, 'fc{}'.format(i), nn.Linear(layers[i], layers[i+1]))
            setattr(self, 'bn{}'.format(i), nn.LayerNorm(layers[i+1]))
        
        
        self.mean = nn.Linear(layers[-2], layers[-1])
        self.var = nn.Linear(layers[-2], layers[-1])

        self.relu = torch.nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(0.2)


    def forward(self, x):
        
        
        for i in range(0,len(self.layers)-2):
            x = getattr(self, 'fc{}'.format(i))(x)
            x = getattr(self, 'bn{}'.format(i))(x)
            x = self.relu(x)
            x = self.dropout(x)
        

        mean = self.mean(x)
        var = self.var(x)
       
        return mean, var

class ConditionDecoder(nn.Module):
    def __init__(self, layers):
        super(ConditionDecoder, self).__init__()
        
        self.layers = layers
        
        # Fully connected layers
        for i in range(len(layers)-1):
            setattr(self, 'fc{}'.format(i), nn.Linear(layers[i], layers[i+1]))
            setattr(self, 'bn{}'.format(i), nn.LayerNorm(layers[i+1]))

        self.relu = torch.nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        for i in range(0,len(self.layers)-1):
            x = getattr(self, 'fc{}'.format(i))(x)
            x = getattr(self, 'bn{}'.format(i))(x)
            x = self.relu(x)
            x = self.dropout(x)

        return x

class ConditionVAE(nn.Module):
    def __init__(self, layers, input_crop):
        super(ConditionVAE, self).__init__()
        self.input_crop = input_crop
        
        layers = [input_crop*128,] + layers
        
        self.encoder = ConditionEncoder(layers)
        self.decoder = ConditionDecoder(layers[::-1])
        self.initialize_weights()
        

    def forward(self, input, encoder_only=False):
        input = input[:,:self.input_crop,:]
        x = input
        
        x = x.view([x.shape[0], -1] )




        mean, var = self.encoder(x)
        if encoder_only: return mean, var
        epsilon = torch.randn_like(var).to(device)        # sampling epsilon        
        z = mean + torch.exp(var / 2) * epsilon          # reparameterization trick
        x_hat = self.decoder(z)
        
        
        
        x = x.view([x.shape[0], self.input_crop, 128])
        x_hat = x_hat.view([x_hat.shape[0], self.input_crop, 128])


        
        return input, x_hat, mean, var


    def initialize_weights(self):
        for layer in list(self.encoder.children()) + list(self.decoder.children()):
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight, gain=0.2)
                nn.init.constant_(layer.bias, 0)  # Initialize biases to zero or another suitable value








class ConditionConvEncoder(nn.Module):
    def __init__(self, channels, linears, input_crop):
        super(ConditionConvEncoder, self).__init__()
        self.input_crop = input_crop
        self.channels = channels
        self.linears = linears


        for i in range(len(channels)-1):
            setattr(self, 'conv{}'.format(i), nn.Conv1d(
                channels[i], channels[i+1], kernel_size=5, stride=2, padding=2
            ))
            setattr(self, 'bnc{}'.format(i), nn.LayerNorm([channels[i+1], input_crop // 2**(i+1)]))
        
        for i in range(len(linears)-2):
            setattr(self, 'fc{}'.format(i), nn.Linear(
                linears[i], linears[i+1],
            ))
            setattr(self, 'bnl{}'.format(i), nn.LayerNorm(linears[i+1]))

        self.mean = nn.Linear(linears[-2], linears[-1])
        self.var = nn.Linear(linears[-2], linears[-1])

        self.relu = torch.nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(0.2)


    def forward(self, x):
        for i in range(0,len(self.channels)-1):
            x = getattr(self, 'conv{}'.format(i))(x)
            x = getattr(self, 'bnc{}'.format(i))(x)
            x = self.relu(x)
            # x = self.dropout(x)
        
        x = x.view([x.shape[0], -1] )
        for i in range(0,len(self.linears)-2):
            x = getattr(self, 'fc{}'.format(i))(x)
            x = getattr(self, 'bnl{}'.format(i))(x)
            x = self.relu(x)
            # x = self.dropout(x)

        mean = self.mean(x)
        var = self.var(x)
       
        return mean, var

class ConditionConvDecoder(nn.Module):
    def __init__(self, channels, linears, input_crop):
        super(ConditionConvDecoder, self).__init__()
        
        self.channels = channels
        self.linears = linears
        self.input_crop = input_crop
        
        # Fully connected layers
        for i in range(len(linears)-1):
            setattr(self, 'fc{}'.format(i), nn.Linear(linears[i], linears[i+1]))
            setattr(self, 'bnl{}'.format(i), nn.LayerNorm(linears[i+1]))

        # make deconvolutional layers
        for i in range(len(self.channels)-1):
            setattr(self, 'deconv{}'.format(i), nn.ConvTranspose1d(
                self.channels[i], self.channels[i+1], kernel_size=5, stride=2, padding=2, output_padding=1
            ))
            setattr(self, 'bnc{}'.format(i), nn.LayerNorm([self.channels[i+1], self.input_crop // 2**(len(self.channels)-2-i)]))




        self.relu = torch.nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        for i in range(0,len(self.linears)-1):
            x = getattr(self, 'fc{}'.format(i))(x)
            x = getattr(self, 'bnl{}'.format(i))(x)
            x = self.relu(x)
            # x = self.dropout(x)

        x = x.view([x.shape[0], self.channels[0], self.input_crop // 2**(len(self.channels)-1)] )

        for i in range(0,len(self.channels)-1):
            x = getattr(self, 'deconv{}'.format(i))(x)
            x = getattr(self, 'bnc{}'.format(i))(x)
            x = self.relu(x)
            # x = self.dropout(x)


        return x

class ConditionConvVAE(nn.Module):
    def __init__(self, channels, linears, input_crop):
        super(ConditionConvVAE, self).__init__()

        intermediate_output_size = int(input_crop / 2**(len(channels)-1)) * channels[-1]
        linears = [intermediate_output_size,] + linears
        self.encoder = ConditionConvEncoder(channels, linears, input_crop)
        self.decoder = ConditionConvDecoder(channels[::-1], linears[::-1], input_crop)
        self.initialize_weights()


        self.input_crop = input_crop
        self.channels = channels
        self.linears = linears
        
        

    def forward(self, input, encoder_only=False):
        input = input[:,:self.input_crop,:]
        x = input
        x = x.swapaxes(1,2) # for making it batch, channels, time
        
        mean, var = self.encoder(x)
        if encoder_only: return mean, var
        epsilon = torch.randn_like(var).to(device)        # sampling epsilon        
        z = mean + torch.exp(var / 2) * epsilon          # reparameterization trick
        x_hat = self.decoder(z)
        
        x = x.swapaxes(1,2) # for making it batch, time, channels
        x_hat = x_hat.swapaxes(1,2) # for making it batch, time, channels
        return input, x_hat, mean, var


    def initialize_weights(self):
        for layer in list(self.encoder.children()) + list(self.decoder.children()):
            if isinstance(layer, nn.Conv1d):
                nn.init.xavier_normal_(layer.weight, gain=0.2)
                nn.init.constant_(layer.bias, 0)  # Initialize biases to zero or another suitable value
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight, gain=0.2)
                nn.init.constant_(layer.bias, 0)  # Initialize biases to zero or another suitable value
