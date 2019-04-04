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
from IPython import embed
from functools import partial
import torch.nn.functional as F
from torch.autograd import Variable


class P3D_module(nn.Module):

    expansion = 4  # global variable for bottleneck

    def __init__(self, params_dict):

        super(P3D_module, self).__init__()
        self.in_planes   = params_dict.get('base_fmaps', None)
        self.out_planes  = params_dict.get('stage_fmaps', None)
        self.downsample  = params_dict.get('downsample', None)
        self.p3d_layers  = params_dict.get('p3d_layers', 13)      # default : 13
        self.cc_type     = params_dict.get('cc_type', 'A')
        self.layer_idx   = params_dict.get('layer_idx', 0)
        self.stride      = params_dict.get('stride', 1)

        if (not self.in_planes) or (not self.out_planes):
            raise TypeError('in_planes and out_planes must be initialized ...')

        self._build_layers()


    def _build_layers(self):
        ''' Build the p3d module '''

        stride_p = self.stride
        print('check stride_p : ', stride_p)
        
        if self.downsample is not None: stride_p = (1, 2, 2)

        if self.layer_idx < self.p3d_layers:

            if self.layer_idx == 0: stride_p = 1

            self.conv1 = nn.Conv3d(self.in_planes, self.out_planes, kernel_size=1, stride=stride_p, bias=False)
            self.bn1   = nn.BatchNorm3d(self.out_planes)

            self.conv2 = self.conv_S(self.out_planes, self.out_planes, stride=1, padding=(0,1,1))
            self.bn2   = nn.BatchNorm3d(self.out_planes)

            self.conv3 = self.conv_T(self.out_planes, self.out_planes, stride=1, padding=(1,0,0))
            self.bn3   = nn.BatchNorm3d(self.out_planes)

            self.conv4 = nn.Conv3d(self.out_planes, self.out_planes*4 , kernel_size=1, bias=False)
            self.bn4   = nn.BatchNorm3d(self.out_planes*4)

        else:

            stride_p = 2 if self.layer_idx == self.p3d_layers else 1
            self.conv1 = nn.Conv2d(self.in_planes, self.out_planes, kernel_size=1, stride=stride_p, bias=False)
            self.bn1   = nn.BatchNorm2d(self.out_planes)

            self.conv_normal = nn.Conv2d(self.out_planes, self.out_planes, kernel_size=3, stride=1, padding=1,bias=False)
            self.bn_normal   = nn.BatchNorm2d(self.out_planes)

            self.conv4 = nn.Conv2d(self.out_planes, self.out_planes*4, kernel_size=1, bias=False)
            self.bn4   = nn.BatchNorm2d(self.out_planes*4)

        self.relu = nn.ReLU(inplace=True)


    def p3d_a(self, x):
        ''' P3D-A block '''

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        return x


    def p3d_b(self, x):
        ''' P3D-B block '''

        temporal_x = self.conv2(x)
        temporal_x = self.bn2(temporal_x)
        temporal_x = self.relu(temporal_x)

        spatial_x = self.conv3(x)
        spatial_x = self.bn3(spatial_x)
        spatial_x = self.relu(spatial_x)

        return spatial_x + temporal_x


    def p3d_c(self, x):
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

        if self.layer_idx < self.p3d_layers:

            if self.cc_type == 'A':
                out = self.p3d_a(out)
            elif self.cc_type == 'B':
                out = self.p3d_b(out)
            elif self.cc_type == 'C':
                out = self.p3d_c(out)
            else:
                raise TypeError('Unknown cc_type, it must be A, B, or C ...')
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

        return nn.Conv3d(in_planes, out_planes, kernel_size=(1,3,3), stride=1, padding=padding, bias=False)


    @staticmethod
    def conv_T(in_planes, out_planes, stride = 1, padding = 1):
        ''' Temporal convolution with filter 3 x 1 x 1 '''

        return nn.Conv3d(in_planes, out_planes, kernel_size=(3,1,1), stride=1, padding=padding, bias=False)
