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


class P3D(nn.Module):

    def __init__(self, block, layers, modality = 'RGB',
        shortcut_type = 'B', num_classes = 400, dropout = 0.5, ST_struc = ('A','B','C')):

        super(P3D, self).__init__()
        self.inplanes = 64
        # self.conv1 = nn.Conv3d(3, 64, kernel_size=7, stride=(1, 2, 2),
        #                        padding=(3, 3, 3), bias=False)
        self.input_channel = 3 if modality=='RGB' else 2  # 2 is for optical flow

        self.ST_struc=ST_struc

        self.conv1_custom = nn.Conv3d(self.input_channel, 64, kernel_size=(1,7,7), stride=(1,2,2),
                                padding=(0,3,3), bias=False)

        self.depth_3d =  sum(layers[:3])# C3D layers are only (res2,res3,res4),  res5 is C2D

        self.bn1  = nn.BatchNorm3d(64) # bn1 is followed by conv1
        self.cnt  = 0
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(2, 3, 3), stride=2, padding=(0,1,1)) # pooling layer for conv1.
        self.maxpool_2 = nn.MaxPool3d(kernel_size=(2,1,1),padding=0,stride=(2,1,1))   # pooling layer for res2, 3, 4.

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


    @property
    def scale_size(self):
        return self.input_size[2] * 256 // 160   # asume that raw images are resized (340,256).

    @property
    def temporal_length(self):
        return self.input_size[1]

    @property
    def crop_size(self):
        return self.input_size[2]

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1):
        downsample = None
        stride_p=stride #especially for downsample branch.

        if self.cnt<self.depth_3d:
            if self.cnt==0:
                stride_p=1
            else:
                stride_p=(1,2,2)
            if stride != 1 or self.inplanes != planes * block.expansion:
                if shortcut_type == 'A':
                    downsample = partial(downsample_basic_block,
                                         planes=planes * block.expansion,
                                         stride=stride)
                else:
                    downsample = nn.Sequential(
                        nn.Conv3d(self.inplanes, planes * block.expansion,
                                  kernel_size=1, stride=stride_p, bias=False),
                        nn.BatchNorm3d(planes * block.expansion)
                    )

        else:
            if stride != 1 or self.inplanes != planes * block.expansion:
                if shortcut_type == 'A':
                    downsample = partial(downsample_basic_block,
                                         planes=planes * block.expansion,
                                         stride=stride)
                else:
                    downsample = nn.Sequential(
                        nn.Conv2d(self.inplanes, planes * block.expansion,
                                  kernel_size=1, stride=2, bias=False),
                        nn.BatchNorm2d(planes * block.expansion)
                    )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample,n_s=self.cnt,depth_3d=self.depth_3d,ST_struc=self.ST_struc))
        self.cnt+=1

        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes,n_s=self.cnt,depth_3d=self.depth_3d,ST_struc=self.ST_struc))
            self.cnt+=1

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1_custom(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.maxpool_2(self.layer1(x))  #  Part Res2
        x = self.maxpool_2(self.layer2(x))  #  Part Res3
        x = self.maxpool_2(self.layer3(x))  #  Part Res4

        sizes=x.size()
        x = x.view(-1,sizes[1],sizes[3],sizes[4])  #  Part Res5
        x = self.layer4(x)
        x = self.avgpool(x)

        x = x.view(-1,self.fc.in_features)
        x = self.fc(self.dropout(x))

        return x
