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




import math




class ConditionConvVAE(nn.Module):
    def __init__(self, channels, linears, input_crop, kernel_size=5, dilation=1, padding=None, stride=2, output_padding=1):
        super(ConditionConvVAE, self).__init__()
        if padding is None:
            padding = kernel_size//2
                 
                 
        # calculate all dimensions here   
        conv_sizes = [input_crop]
        for _ in range(len(channels)-1):
            cs = ( (conv_sizes[-1]+2*padding-dilation*(kernel_size-1)-1)/stride ) + 1
            conv_sizes.append(math.floor(cs))
            print(cs)
        intermediate_output_size = conv_sizes[-1] * channels[-1]
        deconv_sizes = [conv_sizes[-1]]
        for _ in range(len(channels)-1):
            dcs = (deconv_sizes[-1]-1) * stride - 2*padding + dilation * (kernel_size-1) + output_padding + 1
            deconv_sizes.append(dcs)
        
        
        
        linears = [intermediate_output_size,] + linears
        
        

        # encoder
        for i in range(len(channels)-1):
            setattr(self, 'enc_conv{}'.format(i), nn.Conv1d(
                channels[i], channels[i+1], kernel_size=kernel_size, stride=stride, padding=padding
            ))
            setattr(self, 'enc_conv_norm{}'.format(i), nn.LayerNorm([channels[i+1], conv_sizes[i+1]]))
        
        for i in range(len(linears)-2):
            setattr(self, 'enc_lin{}'.format(i), nn.Linear(
                linears[i], linears[i+1],
            ))
            setattr(self, 'enc_lin_norm{}'.format(i), nn.LayerNorm(linears[i+1]))

        self.mean = nn.Linear(linears[-2], linears[-1])
        self.var = nn.Linear(linears[-2], linears[-1])

        dec_linears = linears[::-1]
        dec_channels = channels[::-1]

        # decoder
        # Fully connected layers
        for i in range(len(dec_linears)-1):
            setattr(self, 'dec_lin{}'.format(i), nn.Linear(dec_linears[i], dec_linears[i+1]))
            setattr(self, 'dec_lin_norm{}'.format(i), nn.LayerNorm(dec_linears[i+1]))
            
        # make deconvolutional layers
        for i in range(len(dec_channels)-1):
            setattr(self, 'dec_deconv{}'.format(i), nn.ConvTranspose1d(
                dec_channels[i], dec_channels[i+1], kernel_size=kernel_size, stride=stride, padding=padding, output_padding=output_padding
            ))
            setattr(self, 'dec_deconv_norm{}'.format(i), nn.LayerNorm([dec_channels[i+1], deconv_sizes[i+1]]))
    
        self.relu = torch.nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(0.2)

        self.linears = linears
        self.channels = channels
        self.dec_linears = dec_linears
        self.dec_channels = dec_channels
        self.conv_sizes = conv_sizes
        self.deconv_sizes = deconv_sizes
        
        print('conv sizes', conv_sizes)
        print('linears', linears)
        print('dec linears', dec_linears)
        print('dec deconvs', deconv_sizes)
        
        self.initialize_weights()

        self.channels = channels
        self.linears = linears
        self.input_crop = input_crop
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.padding = padding
        self.stride = stride
        self.output_padding = output_padding

    def forward(self, input, encoder_only=False):
        mean, var = self.encode(input)

        # reparameterization trick
        if encoder_only: return mean, var
        epsilon = torch.randn_like(var).to(device)        # sampling epsilon        
        x = mean + torch.exp(var / 2) * epsilon          # reparameterization trick
        
        x = self.decode(x)
        
        return x, mean, var


    def encode(self, x):
        x = x[:,:self.input_crop,:]
        x = x.swapaxes(1,2) # for making it batch, channels, time
        
        # encoder
        for i in range(0,len(self.channels)-1):
            x = getattr(self, 'enc_conv{}'.format(i))(x)
            x = getattr(self, 'enc_conv_norm{}'.format(i))(x)
            x = self.relu(x)
            x = self.dropout(x)
        
        x = x.view([x.shape[0], -1] )
        for i in range(0,len(self.linears)-2):
            x = getattr(self, 'enc_lin{}'.format(i))(x)
            x = getattr(self, 'enc_lin_norm{}'.format(i))(x)
            x = self.relu(x)
            x = self.dropout(x)

        mean = self.mean(x)
        var = self.var(x)
       
        return mean, var

    def decode(self, x):  
        # decoder
        for i in range(0,len(self.linears)-1):
            x = getattr(self, 'dec_lin{}'.format(i))(x)
            x = getattr(self, 'dec_lin_norm{}'.format(i))(x)
            x = self.relu(x)
            x = self.dropout(x)

        x = x.view([x.shape[0], self.dec_channels[0], self.deconv_sizes[0]]) # batch, channels, time

        for i in range(0,len(self.channels)-1):
            x = getattr(self, 'dec_deconv{}'.format(i))(x)
            x = getattr(self, 'dec_deconv_norm{}'.format(i))(x)
            x = self.relu(x)
            x = self.dropout(x)

        x = x[:,:,:self.input_crop] # crop it to input crop for calculating loss etc
        x = x.swapaxes(1,2) # for making it batch, time, channels
        
        return x

    def initialize_weights(self):
        for layer in list(self.children()):
            if isinstance(layer, nn.Conv1d):
                nn.init.xavier_normal_(layer.weight, gain=1)
                nn.init.constant_(layer.bias, 0)  # Initialize biases to zero or another suitable value
            if isinstance(layer, nn.ConvTranspose1d):
                nn.init.xavier_normal_(layer.weight, gain=1)
                nn.init.constant_(layer.bias, 0)  # Initialize biases to zero or another suitable value
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight, gain=1)
                nn.init.constant_(layer.bias, 0)  # Initialize biases to zero or another suitable value

