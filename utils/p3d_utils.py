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
from IPython import embed
from torch.autograd import Variable
from utils.p3d_module import P3D_module

times = 4

class P3D(nn.Module):

    def __init__(self, params_dict):

        super(P3D, self).__init__()
        self.p3d_module  = params_dict.pop('p3d_module', P3D_module)
        self.layers_list = params_dict.pop('layers_list', [3, 4, 6, 3])  # default ResNet-50
        self.cc_type     = params_dict.pop('cc_type', 'A')
        self.n_classes   = params_dict.pop('n_classes', 101)
        self.in_channel  = params_dict.pop('in_channel', 3)
        self.base_fmaps  = params_dict.pop('base_fmaps', 64)
        self.layer_idx   = params_dict.pop('layer_idx', 0)
        self.p_dropout   = params_dict.pop('p_dropout', 0.5)

        self._build_layers()


    def _build_layers(self):
        ''' Build layers according to the style of resnet '''

        self.p3d_layers  = sum(self.layers_list[:3])
        self.times   = times
        self.maxpool = nn.MaxPool3d(kernel_size=(2,1,1), stride=(2,1,1), padding=0)
        self.avgpool = nn.AvgPool2d(kernel_size=(4,4), stride=1)    # default (5, 5) for (160, 160)-input
        self.dropout = nn.Dropout(p=self.p_dropout)

        self.pre = nn.Sequential(
             nn.Conv3d(self.in_channel, 64, kernel_size=(1,7,7), stride=(1,2,2), padding=(0,3,3), bias=False),
             nn.BatchNorm3d(64),
             nn.ReLU(inplace=True),
             nn.MaxPool3d(kernel_size=(2,3,3), stride=2, padding=(0,1,1))
        )
        self.layer1 = self._make_layer(64,  self.layers_list[0])
        self.layer2 = self._make_layer(128, self.layers_list[1], stride=2)
        self.layer3 = self._make_layer(256, self.layers_list[2], stride=2)
        self.layer4 = self._make_layer(512, self.layers_list[3], stride=2)
        self.fc     = nn.Linear(512*self.times, self.n_classes)

        # Xavier-init
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        # some private attribute
        self.input_size = (self.in_channel, 16, 160, 160)
        self.input_mean = [0.485, 0.456, 0.406] if self.in_channel == 3 else [0.5000]
        self.input_std  = [0.229, 0.224, 0.225] if self.in_channel == 3 else [0.2260]  # np.mean([0.229, 0.224, 0.225])


    def _downsample(self, stage_fmaps, stride = 1):
        ''' Generate the downsample method for following _make_layer '''

        downsample = None

        if self.layer_idx < self.p3d_layers:

            stride_p = 1 if self.layer_idx == 0 else (1, 2, 2)

            if (stride != 1) or (self.base_fmaps != stage_fmaps*self.times):
                if self.cc_type == 'A':
                    downsample = partial(self.downsample_basic_block, planes=stage_fmaps*self.times, stride=stride)
                else:
                    downsample = nn.Sequential(
                        nn.Conv3d(self.base_fmaps, stage_fmaps*self.times, kernel_size=1, stride=stride_p, bias=False),
                        nn.BatchNorm3d(stage_fmaps*self.times)
                    )
        else:
            if (stride != 1) or (self.base_fmaps != stage_fmaps*self.times):

                if self.cc_type == 'A':
                    downsample = partial(self.downsample_basic_block, planes=stage_fmaps*self.times, stride=stride)
                else:
                    downsample = nn.Sequential(
                        nn.Conv2d(self.base_fmaps, stage_fmaps*self.times, kernel_size=1, stride=2, bias=False),
                        nn.BatchNorm2d(stage_fmaps*self.times)
                    )
        return downsample


    def _make_layer(self, stage_fmaps, n_layers, stride = 1):
        '''
        Generate the ResidualModule

        step - 1. prepare the downsample for first-layer of each residual modules
        step - 2. build the first layer for residual block
        step - 3. update the mid-variable
        step - 4. build the remain layers for residual block
        '''

        stride_p, layers = stride, []

        # step - 1
        downsample = self._downsample(stage_fmaps, stride)

        # step - 2
        params_dict = {
            'p3d_layers'  : self.p3d_layers,
            'base_fmaps'  : self.base_fmaps,
            'layer_idx'   : self.layer_idx,
            'cc_type'     : self.cc_type,
            'stage_fmaps' : stage_fmaps,
            'downsample'  : downsample,
            'stride'      : stride
        }
        layers.append(self.p3d_module(params_dict))

        # step - 3
        self.layer_idx += 1
        self.base_fmaps = stage_fmaps * self.times

        # step - 4
        del params_dict['stride']
        del params_dict['downsample']
        for i in range(1, n_layers):
            params_dict['layer_idx'] = self.layer_idx
            layers.append(self.p3d_module(params_dict))
            self.layer_idx += 1

        return nn.Sequential(*layers)


    def forward(self, x):

        print('input size : ', x.shape)

        x = self.pre(x)
        print('after pre : ', x.shape)

        x = self.maxpool(self.layer1(x))  #  Part Res2
        print('after layer1 : ', x.shape)

        x = self.maxpool(self.layer2(x))  #  Part Res3
        print('after layer2 : ', x.shape)

        x = self.maxpool(self.layer3(x))  #  Part Res4
        print('after layer3 : ', x.shape)

        sizes = x.size()
        x = x.view(-1,sizes[1],sizes[3],sizes[4])  #  Part Res5
        print('before layer4, first view : ', x.shape)

        x = self.layer4(x)
        print('after layer4 : ', x.shape)

        x = self.avgpool(x)
        print('avgpool, (5, 5) : ', x.shape)

        x = x.view(-1,self.fc.in_features)
        print('before fc, second view : ', x.shape)

        x = self.fc(self.dropout(x))
        print('after fc : ', x.shape)


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

    @staticmethod
    def downsample_basic_block(x, planes, stride):
        ''' Padding in temporal dimension '''
        out = F.avg_pool3d(x, kernel_size=1, stride=stride)
        zero_pads = torch.Tensor(out.size(0), planes - out.size(1), out.size(2), \
                                 out.size(3), out.size(4)).zero_()

        # if isinstance(out.data, torch.cuda.FloatTensor):
        #     zero_pads = zero_pads.cuda()

        out = Variable(torch.cat([out.data, zero_pads], dim=1))

        return out


def P3D_zoo(params_dict):
    ''' Choose P3D-model from P3D63, P3D131, or P3D199 ... '''

    model_type = params_dict.get('ResNet', 50)

    if model_type == 50:
        params_dict['layers_list'] = [3, 4, 6, 3]
    elif model_type == 101:
        params_dict['layers_list'] = [3, 4, 23, 3]
    elif model_type == 151:
        params_dict['layers_list'] = [3, 8, 36, 3]
    else:
        raise TypeError('Unknown ResNet-id, it should be 50, 101, or 151 ..')

    if params_dict.get('source', 'rgb') == 'rgb':
        params_dict['in_channel'] = 3
    else:
        params_dict['in_channel'] = 2

    model = P3D(params_dict)

    if params_dict.get('pretrained', False):
        weights = torch.load(params_dict['init_dir'])['state_dict']
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
