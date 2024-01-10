import torch
import torch.nn as nn


import random
import numpy as np


device = 'cuda'
seed = 1234

torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)



class ConditionEncoder(nn.Module):
    def __init__(self):
        super(ConditionEncoder, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, stride=2, padding=2)
        self.conv2 = nn.Conv2d(32, 16, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(16, 8, kernel_size=3, stride=2, padding=1)
        # self.conv4 = nn.Conv2d(8, 4, kernel_size=3, stride=1, padding=1)

        # Fully connected layers
        self.fc1 = nn.Linear(16*16*8, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 32)
        self.fc4 = nn.Linear(32, 2)

    def forward(self, x):
        x = x.view([x.shape[0], 1, x.shape[1], x.shape[2]] )
        # Convolutional layers with ReLU activation and max pooling
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        # x = self.pool(torch.relu(self.conv4(x)))
        # Flatten the output for the fully connected layers
        x = x.view(-1, 16*16*8)
        # Fully connected layers with ReLU activation
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.tanh(self.fc4(x))
        return x

class ConditionDecoder(nn.Module):
    def __init__(self):
        super(ConditionDecoder, self).__init__()
        
        # Fully connected layers
        self.fc1 = nn.Linear(2, 32)
        self.fc2 = nn.Linear(32, 128)
        self.fc3 = nn.Linear(128, 512)
        self.fc4 = nn.Linear(512, 16*16*8)

        
        # Deconvolutional layers
        
        
        self.deconv1 = nn.ConvTranspose2d(8, 16, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv2 = nn.ConvTranspose2d(16, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv3 = nn.ConvTranspose2d(32, 1, kernel_size=5, stride=2, padding=2, output_padding=1)
        # self.conv4 = nn.Conv2d(8, 4, kernel_size=3, stride=1, padding=1)

    def forward(self, x):

        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        

        x = x.view([x.shape[0], 8, 16, 16])

        # Convolutional layers with ReLU activation and max pooling
        x = torch.relu(self.deconv1(x))
        x = torch.relu(self.deconv2(x))
        x = torch.tanh(self.deconv3(x))

        return x

class ConditionVA(nn.Module):
    def __init__(self):
        super(ConditionVA, self).__init__()
        self.encoder = ConditionEncoder()
        self.decoder = ConditionDecoder()
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


