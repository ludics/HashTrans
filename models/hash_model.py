import torch
import torch.nn as nn
import numpy as np

from .build import build_model


class DSHNet(nn.Module):
    def __init__(self, config, pretrained=True):
        super(DSHNet, self).__init__()

        self.swin = build_model(config)
        ckpt = torch.load(config.MODEL.RESUME, map_location='cpu')
        msg = self.swin.load_state_dict(ckpt['model'], strict=False)
        del ckpt
        self.hash_layer = nn.Linear(self.swin.num_features, config.HASH_BIT)
        
    def forward(self, x):
        feats = self.swin.forward_features(x)
        y = self.hash_layer(feats)
        return y

