import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import timm

from .build import build_model


class DSHNet(nn.Module):
    def __init__(self, config, pretrained=True):
        super(DSHNet, self).__init__()
        if config.MODEL.TYPE == 'resnet':
            self.resnet = timm.create_model(config.MODEL.NAME, pretrained=True)
            self.num_features = self.resnet.num_features
        elif config.MODEL.TYPE == 'swin':
            self.swin = build_model(config)
            ckpt = torch.load(config.HASH.PRETRAINED, map_location='cpu')
            del ckpt['model']['head.weight']
            del ckpt['model']['head.bias']
            msg = self.swin.load_state_dict(ckpt['model'], strict=False)
            del ckpt
            self.num_features = self.swin.num_features
        elif config.MODEL.TYPE == 'swin_gwl':
            self.swin = build_model(config)
            self.num_features = self.swin.num_features
        self.model_type = config.MODEL.TYPE
        self.hash_layer = nn.Linear(self.num_features, config.HASH.HASH_BIT)
        # self.cls_head = nn.Linear(self.num_features, config.MODEL.NUM_CLASSES)
        self.cls_head = nn.Linear(config.HASH.HASH_BIT, config.MODEL.NUM_CLASSES)
    
    def feat_extract(self, x):
        if self.model_type == 'swin':
            return self.swin.forward_features(x)
        elif self.model_type == 'swin_gwl':
            x, sp_v, ch_v = self.swin.forward_features(x)
            return x, sp_v, ch_v
        elif self.model_type == 'resnet':
            x = self.resnet.forward_features(x)
            x = self.resnet.global_pool(x)
            if self.resnet.drop_rate:
                x = F.dropout(x, p=float(self.drop_rate), training=self.training)
            return x

    def forward(self, x):
        if self.model_type == 'swin_gwl':
            feats, sp_v, ch_v = self.feat_extract(x)
            hash_bits = self.hash_layer(feats)
            preds = self.cls_head(hash_bits)
            if self.training:
                return hash_bits, preds, sp_v, ch_v
            else:
                return hash_bits, preds
        else:
            feats = self.feat_extract(x)
            hash_bits = self.hash_layer(feats)
            preds = self.cls_head(hash_bits)
            return hash_bits, preds

