#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2019/04/02
author: lujie
"""

import torch
from utils.p3d_utils import P3D
from utils.bottleneck_utils import Bottleneck


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
