#!/usr/bin/env python3

import torch
import torch.nn as nn

class TNET(nn.Module):
    def __init__(self, fc_size):
        super(TNET, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(6, 64, 3, 2),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(64, 128, 3, 2),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(128, 256, 3, 2),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(256, 512, 3, 2),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(512, 128, 1, 1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Fully Connected layers
        self.fc = nn.Sequential(
            nn.Linear(fc_size, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.1),
            nn.Linear(16, 3),
            nn.LeakyReLU(0.2, inplace=True))

    def forward(self, x):
        x = self.cnn(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x