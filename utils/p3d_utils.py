#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2019/04/02
author: lujie
"""

import math
import torch
import numpy as np
import torch.nn as nn
from functools import partial
import torch.nn.functional as F
from torch.autograd import Variable
from utils.bottleneck_utils import Bottleneck

class P3D(nn.Module):

    def __init__(self, block, layers, modality = 'RGB',
        shortcut_type = 'B', num_classes = 400, dropout = 0.5, ST_struc = ('A','B','C')):

        super(P3D, self).__init__()
        self.inplanes = 64
        self.input_channel = 3 if modality=='RGB' else 2  # 2 is for optical flow
        self.depth_3d  =  sum(layers[:3])# C3D layers are only (res2,res3,res4),  res5 is C2D
        self.ST_struc  = ST_struc
        self.layer_idx = 0

        self.pre = nn.Sequential(
             nn.Conv3d(self.input_channel, 64, kernel_size=(1,7,7), stride=(1,2,2), padding=(0,3,3), bias=False),
             nn.BatchNorm3d(64),
             nn.ReLU(inplace=True)
             nn.MaxPool3d(kernel_size=(2, 3, 3), stride=2, padding=(0,1,1))
        )

        self.maxpool = nn.MaxPool3d(kernel_size=(2,1,1), stride=(2,1,1), padding=0)

        self.layer1 = self._make_layer(block, 64,  layers[0], shortcut_type)
        self.layer2 = self._make_layer(block, 128, layers[1], shortcut_type, stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], shortcut_type, stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], shortcut_type, stride=2)

        self.avgpool = nn.AvgPool2d(kernel_size=(5, 5), stride=1)                              # pooling layer for res5.
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        # some private attribute
        self.input_size = (self.input_channel,16,160,160)       # input of the network
        self.input_mean = [0.485, 0.456, 0.406] if modality=='RGB' else [0.5]
        self.input_std  = [0.229, 0.224, 0.225] if modality=='RGB' else [np.mean([0.229, 0.224, 0.225])]


    def _make_layer(self, block, planes, num_blocks, shortcut_type, stride=1):
        '''
        Generate the ResidualBlock

        step - 1. prepare the downsample
        step - 2. generate the layers
        '''

        downsample, stride_p, layers = None, stride, []

        # step - 1
        if self.layer_idx < self.depth_3d:

            stride_p = 1 if self.layer_idx == 0 else (1, 2, 2)

            if (stride != 1) or (self.inplanes != planes * block.expansion):
                if shortcut_type == 'A':
                    downsample = partial(block.downsample_basic_block, planes=planes * block.expansion, stride=stride)
                else:
                    downsample = nn.Sequential(
                        nn.Conv3d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride_p, bias=False),
                        nn.BatchNorm3d(planes * block.expansion)
                    )
        else:
            if (stride != 1) or (self.inplanes != planes * block.expansion):

                if shortcut_type == 'A':
                    downsample = partial(block.downsample_basic_block, planes=planes * block.expansion, stride=stride)
                else:
                    downsample = nn.Sequential(
                        nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=2, bias=False),
                        nn.BatchNorm2d(planes * block.expansion)
                    )

        layers.append(block(self.inplanes, planes, stride, downsample, layer_idx=self.layer_idx, \
                                depth_3d=self.depth_3d, ST_struc=self.ST_struc))
        self.layer_idx += 1

        self.inplanes = planes * block.expansion

        for i in range(1, num_blocks):
            layers.append(block(self.inplanes, planes, layer_idx=self.layer_idx, depth_3d=self.depth_3d, ST_struc=self.ST_struc))
            self.layer_idx += 1

        return nn.Sequential(*layers)


    def forward(self, x):

        x = self._pre(x)
        x = self.maxpool(self.layer1(x))  #  Part Res2
        x = self.maxpool(self.layer2(x))  #  Part Res3
        x = self.maxpool(self.layer3(x))  #  Part Res4

        sizes = x.size()
        x = x.view(-1,sizes[1],sizes[3],sizes[4])  #  Part Res5
        x = self.layer4(x)
        x = self.avgpool(x)

        x = x.view(-1,self.fc.in_features)
        x = self.fc(self.dropout(x))

        return x


    @property
    def scale_size(self):
        return self.input_size[2] * 256 // 160   # asume that raw images are resized (340,256).


    @property
    def temporal_length(self):
        return self.input_size[1]


    @property
    def crop_size(self):
        return self.input_size[2]



def P3D63(**kwargs):
    ''' Construct a P3D63 modelbased on a ResNet-50-3D model '''

    model = P3D(Bottleneck, [3, 4, 6, 3], **kwargs)

    return model


def P3D131(**kwargs):
    ''' Construct a P3D131 model based on a ResNet-101-3D model '''

    model = P3D(Bottleneck, [3, 4, 23, 3], **kwargs)

    return model


def P3D199(pretrained = False, modality = 'RGB', **kwargs):
    ''' Construct a P3D199 model based on a ResNet-152-3D model '''

    model = P3D(Bottleneck, [3, 8, 36, 3], modality=modality, **kwargs)

    if pretrained == True:
        if modality == 'RGB':
            pretrained_file = 'p3d_rgb_199.checkpoint.pth.tar'
        elif modality == 'Flow':
            pretrained_file = 'p3d_flow_199.checkpoint.pth.tar'
        weights = torch.load(pretrained_file)['state_dict']
        model.load_state_dict(weights)

    return model


# custom operation
def get_optim_policies(model = None, modality = 'RGB', enable_pbn = True):
    '''
    first conv:         weight --> conv weight
                        bias   --> conv bias
    normal action:      weight --> non-first conv + fc weight
                        bias   --> non-first conv + fc bias
    bn:                 the first bn2, and many all bn3.

    '''
    first_conv_weight = []
    first_conv_bias = []
    normal_weight = []
    normal_bias = []
    bn = []

    if model == None:
        log.l.info('no model!')
        exit()

    conv_cnt = 0
    bn_cnt = 0
    for m in model.modules():
        if isinstance(m, torch.nn.Conv3d) or isinstance(m, torch.nn.Conv2d):
            ps = list(m.parameters())
            conv_cnt += 1
            if conv_cnt == 1:
                first_conv_weight.append(ps[0])
                if len(ps) == 2:
                    first_conv_bias.append(ps[1])
            else:
                normal_weight.append(ps[0])
                if len(ps) == 2:
                    normal_bias.append(ps[1])
        elif isinstance(m, torch.nn.Linear):
            ps = list(m.parameters())
            normal_weight.append(ps[0])
            if len(ps) == 2:
                normal_bias.append(ps[1])

        elif isinstance(m, torch.nn.BatchNorm3d):
            bn_cnt += 1
            # later BN's are frozen
            if not enable_pbn or bn_cnt == 1:
                bn.extend(list(m.parameters()))
        elif isinstance(m,torch.nn.BatchNorm2d):
            bn.extend(list(m.parameters()))
        elif len(m._modules) == 0:
            if len(list(m.parameters())) > 0:
                raise ValueError("New atomic module type: {}. Need to give it a learning policy".format(type(m)))

    slow_rate=0.7
    n_fore=int(len(normal_weight)*slow_rate)
    slow_feat=normal_weight[:n_fore] # finetune slowly.
    slow_bias=normal_bias[:n_fore]
    normal_feat=normal_weight[n_fore:]
    normal_bias=normal_bias[n_fore:]

    return [
        {'params': first_conv_weight, 'lr_mult': 5 if modality == 'Flow' else 1, 'decay_mult': 1,
         'name': "first_conv_weight"},
        {'params': first_conv_bias, 'lr_mult': 10 if modality == 'Flow' else 2, 'decay_mult': 0,
         'name': "first_conv_bias"},
        {'params': slow_feat, 'lr_mult': 1, 'decay_mult': 1,
         'name': "slow_feat"},
        {'params': slow_bias, 'lr_mult': 2, 'decay_mult': 0,
         'name': "slow_bias"},
        {'params': normal_feat, 'lr_mult': 1 , 'decay_mult': 1,
         'name': "normal_feat"},
        {'params': normal_bias, 'lr_mult': 2, 'decay_mult':0,
         'name': "normal_bias"},
        {'params': bn, 'lr_mult': 1, 'decay_mult': 0,
         'name': "BN scale/shift"},
    ]
