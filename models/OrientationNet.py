import torch
import torch.nn as nn

from .FlowNetS import *


class ONET(nn.Module):
    def __init__(self, fc_size):
        super(ONET, self).__init__()

        self.flownet = FlowNetS(batchNorm=False, input_channels=6)

        # Fully Connected layers
        self.fc = nn.Sequential(
            nn.LeakyReLU(0,2, inplace=True),
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
        x = self.flownet(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x
