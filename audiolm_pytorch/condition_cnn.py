import torch
import torch.nn as nn


import random
import numpy as np


device = 'cuda'
seed = 1234

torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

class ConditionCNN(nn.Module):
    def __init__(self, pre_layers, post_layers, dropout_rate):
        super(ConditionCNN, self).__init__()
        # Fully connected layers
        
        self.pre_layers = pre_layers
        self.post_layers = post_layers

        for i in range(len(self.pre_layers)-1):
            setattr(self, 'fc1{}'.format(i+1), nn.Linear(pre_layers[i], pre_layers[i+1]))
            setattr(self, 'bn1{}'.format(i+1), nn.BatchNorm1d(pre_layers[i+1]))
        
        self.bottleneck = nn.Linear(pre_layers[-1],post_layers[0])

        for i in range(len(self.post_layers)-2):
            setattr(self, 'fc2{}'.format(i+1), nn.Linear(post_layers[i], post_layers[i+1]))
            setattr(self, 'bn2{}'.format(i+1), nn.BatchNorm1d(post_layers[i+1]))
        
        self.out = nn.Linear(post_layers[-2],post_layers[-1])

        # self.relu = torch.nn.LeakyReLU(0.2)
        self.relu = torch.nn.Tanh()
        self.dropout = nn.Dropout(dropout_rate)

        self.initialize_weights()



    def forward(self, x, bn_only=False):
        x = x.view([x.shape[0], -1] )
        x = self.dropout(x)

        for i in range(1,len(self.pre_layers)):
            x = getattr(self, 'fc1{}'.format(i))(x)
            x = getattr(self, 'bn1{}'.format(i))(x)
            x = self.relu(x)
            if i < len(self.pre_layers)-0:
                x = self.dropout(x)
        

        x = self.bottleneck(x)
        bn = x

        if bn_only:
            return bn

        for i in range(1,len(self.post_layers)-2):
            x = getattr(self, 'fc2{}'.format(i))(x)
            x = getattr(self, 'bn2{}'.format(i))(x)
            x = self.relu(x)
            # x = self.dropout(x)
        
        out = self.out(x)
       
        return out, bn

    def initialize_weights(self):
        for layer in list(self.children()):
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight, gain=1.0)
                nn.init.constant_(layer.bias, 0)  # Initialize biases to zero or another suitable value
