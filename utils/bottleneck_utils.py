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


class Bottleneck(nn.Module):

    def __init__(self, inplanes, planes, stride = 1, downsample = None, n_s = 0, depth_3d = 47, ST_struc = ('A','B','C') ):

        super(Bottleneck, self).__init__()
        self.downsample = downsample
        self.depth_3d   = depth_3d
        self.ST_struc   = ST_struc
        self.len_ST     = len(self.ST_struc)

        stride_p = stride
        if not self.downsample == None:
            stride_p = (1, 2, 2)

        if n_s < self.depth_3d:
            if n_s == 0:
                stride_p = 1
            self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False,stride=stride_p)
            self.bn1   = nn.BatchNorm3d(planes)
        else:
            if n_s == self.depth_3d:
                stride_p = 2
            else:
                stride_p = 1
            self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False,stride=stride_p)
            self.bn1   = nn.BatchNorm2d(planes)
        # self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=stride,
        #                        padding=1, bias=False)
        self.id = n_s
        self.ST = list(self.ST_struc)[self.id%self.len_ST]
        if self.id < self.depth_3d:
            self.conv2 = conv_S(planes,planes, stride=1,padding=(0,1,1))
            self.bn2   = nn.BatchNorm3d(planes)
            #
            self.conv3 = conv_T(planes,planes, stride=1,padding=(1,0,0))
            self.bn3   = nn.BatchNorm3d(planes)
        else:
            self.conv_normal = nn.Conv2d(planes, planes, kernel_size=3, stride=1,padding=1,bias=False)
            self.bn_normal   = nn.BatchNorm2d(planes)

        if n_s < self.depth_3d:
            self.conv4 = nn.Conv3d(planes, planes * 4, kernel_size=1, bias=False)
            self.bn4   = nn.BatchNorm3d(planes * 4)
        else:
            self.conv4 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
            self.bn4   = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)

        self.stride = stride


    def ST_A(self, x):
        ''' P3D-A block '''

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        return x


    def ST_B(self, x):
        ''' P3D-B block '''

        temporal_x = self.conv2(x)
        temporal_x = self.bn2(temporal_x)
        temporal_x = self.relu(temporal_x)

        spatial_x = self.conv3(x)
        spatial_x = self.bn3(spatial_x)
        spatial_x = self.relu(spatial_x)

        return spatial_x + temporal_x


    def ST_C(self, x):
        ''' P3D-C block '''

        spatial_x = self.conv2(x)
        spatial_x = self.bn2(spatial_x)
        spatial_x = self.relu(spatial_x)

        temporal_x = self.conv3(spatial_x)
        temporal_x = self.bn3(temporal_x)
        temporal_x = self.relu(temporal_x)

        return spatial_x + temporal_x


    def forward(self, x):
        ''' forward '''

        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # out = self.conv2(out)
        # out = self.bn2(out)
        # out = self.relu(out)

        if self.id < self.depth_3d: # C3D parts:
            if self.ST == 'A':
                out = self.ST_A(out)
            elif self.ST == 'B':
                out = self.ST_B(out)
            elif self.ST == 'C':
                out = self.ST_C(out)
        else:
            out = self.conv_normal(out)   # normal is res5 part, C2D all.
            out = self.bn_normal(out)
            out = self.relu(out)

        out = self.conv4(out)
        out = self.bn4(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


    @staticmethod
    def conv_S(in_planes, out_planes, stride = 1, padding = 1):
        ''' Spatial convolution with filter 1 x 3 x 3 '''

        return nn.Conv3d(in_planes,out_planes,kernel_size=(1,3,3),stride=1,
                         padding=padding,bias=False)


    @staticmethod
    def conv_T(in_planes, out_planes, stride = 1, padding = 1):
        ''' Temporal convolution with filter 3 x 1 x 1 '''

        return nn.Conv3d(in_planes,out_planes,kernel_size=(3,1,1),stride=1,
                         padding=padding,bias=False)


    @staticmethod
    def downsample_basic_block(x, planes, stride):
        '''  '''
        out = F.avg_pool3d(x, kernel_size=1, stride=stride)
        zero_pads = torch.Tensor(out.size(0), planes - out.size(1),
                                 out.size(2), out.size(3),
                                 out.size(4)).zero_()
        if isinstance(out.data, torch.cuda.FloatTensor):
            zero_pads = zero_pads.cuda()

        out = Variable(torch.cat([out.data, zero_pads], dim=1))

        return out
