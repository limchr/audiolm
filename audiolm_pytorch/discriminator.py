import torch
import torch.nn as nn

import math
import random
import numpy as np

import torch.nn.functional as F



class Discriminator(nn.Module):
    def __init__(self, channels, linears, input_crop, dropout):
        super(Discriminator, self).__init__()

        intermediate_output_size = math.ceil(input_crop / 2**(len(channels)-1)) * channels[-1]
        conv_sizes = [input_crop,] + [math.ceil(input_crop / 2**(i+1)) for i in range(len(channels)-1)]
        linears = [intermediate_output_size,] + linears
        print("creating discriminator with linears", linears)
        
        for i in range(len(channels)-1):
            setattr(self, 'conv{}'.format(i), nn.Conv1d(
                channels[i], channels[i+1], kernel_size=5, stride=2, padding=2
            ))
            setattr(self, 'bnc{}'.format(i), nn.LayerNorm([channels[i+1], conv_sizes[i+1]]))
        
        for i in range(len(linears)-2):
            setattr(self, 'fc{}'.format(i), nn.Linear(
                linears[i], linears[i+1],
            ))
            setattr(self, 'bnl{}'.format(i), nn.LayerNorm(linears[i+1]))

        self.out = nn.Linear(linears[-2], linears[-1])

        self.relu = torch.nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)

        self.initialize_weights()

        self.input_crop = input_crop
        self.channels = channels
        self.linears = linears




    def forward(self, x, softmax):
        x = x[:,:self.input_crop,:]
        x = x.swapaxes(1,2) # for making it batch, channels, time
        
        
        for i in range(0,len(self.channels)-1):
            x = getattr(self, 'conv{}'.format(i))(x)
            x = getattr(self, 'bnc{}'.format(i))(x)
            x = self.relu(x)
            x = self.dropout(x)
        
        x = x.view([x.shape[0], -1] )
        for i in range(0,len(self.linears)-2):
            x = getattr(self, 'fc{}'.format(i))(x)
            x = getattr(self, 'bnl{}'.format(i))(x)
            x = self.relu(x)
            x = self.dropout(x)

        out = self.out(x)
        if softmax:
            out = F.softmax(out, dim=1)
       
        return out

    def initialize_weights(self):
        for layer in self.children():
            if isinstance(layer, nn.Conv1d):
                nn.init.xavier_normal_(layer.weight, gain=0.2)
                nn.init.constant_(layer.bias, 0)  # Initialize biases to zero or another suitable value
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight, gain=0.2)
                nn.init.constant_(layer.bias, 0)  # Initialize biases to zero or another suitable value
