#!/usr/bin/env python3

import torch
import torch.nn as nn
from .OrientationNet import ONET
from .TranslationNet import TNET

class Criterion(nn.Module):
    def __init__(self, orientation='euler', k=100.0):
        super(Criterion, self).__init__()
        self.orientation = orientation
        self.k = k
    
    def _normalize_quaternion(self, q):
        magnitude = torch.norm(q)
        q = torch.div(q, magnitude)

        return q

    def forward(self, pred, gt):
        p_hat, p = pred[:, :3], gt[:, :3] # translation
        q_hat, q = pred[:, 3:], gt[:, 3:] # orientation(euler)
        
        if self.orientation == 'quaternion':
            q_hat = self._normalize_quaternion(q_hat)
        
        p_error = nn.MSELoss()(p, p_hat)
        q_error = nn.MSELoss()(q, q_hat)

        loss = p_error + (self.k * q_error)
        
        return loss
    
class DeepPCO(nn.Module):
    def __init__(self):
        super(DeepPCO, self).__init__()

        # self.t_coeff = nn.Parameter(torch.Tensor([init_t_coeff]))
        # self.o_coeff = nn.Parameter(torch.Tensor([init_o_coeff]))

        self.onet = ONET(fc_size=8192)
        self.tnet = TNET(fc_size=24192)

    def load_Flownet(self, flownet_dir):
        pretrained_w = torch.load(flownet_dir, map_location='cpu')

        # Load FlowNet weights pretrained with FlyingChairs
        # NOTE: the pretrained model assumes image rgb values in range [-0.5, 0.5]

        # Use only conv-layer of FlowNet as CNN for DeepVO
        model_dict = self.state_dict()
        update_dict = {k: v for k, v in pretrained_w['state_dict'].items() if k in model_dict}
        model_dict.update(update_dict)
        self.load_state_dict(model_dict)
    
    def forward(self, x):
        translation = self.tnet(x)
        orientation = self.onet(x)

        pose = torch.cat((translation, orientation), dim=1) # (x, y, z, roll, pitch, yaw)
        return pose
