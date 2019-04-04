#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2019/04/02
author: lujie
"""

import torch
from utils.p3d_utils import P3D_zoo
from torchsummary import summary
from IPython import embed


if __name__ == '__main__':

    params_dict = {
        'ResNet'    : 50,
        'init_dir'  : None,   # pretrained_file path
        'pretrained': False,
        'source'    : 'rgb',
        'n_classes' : 101,
        'cc_type'   : 'A',    # P3D-type
        'p_dropout' : 0.5
    }


    model = P3D_zoo(params_dict)
    # embed()
    # summary(model, (3, 16, 224, 224))
    # model = model.cuda()
    # data = torch.autograd.Variable(torch.rand(1,3, 16, 128, 128))  # if modality=='Flow', please change the 2nd dimension 3==>2
    data = torch.autograd.Variable(torch.rand(1,3,16,112,112))
    out = model(data)
    print(out.shape)
