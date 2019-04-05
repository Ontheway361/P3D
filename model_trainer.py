#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2019/04/02
author: lujie
"""

import torch
from runner.p3d_runner import P3D_Train


if __name__ == '__main__':

    params_dict = {

        'model_info'  : {
             'ResNet'      : 50,
             'source'      : 'rgb',
             'num_classes' : 101,
             'pretrained'  : False,  # NO-USE
             'p_dropout'   : 0.5,
             'cc_type'     : 'C',    # P3D-type
        },
        'dataset'     : 'ucf101',
        'clip_len'    : 16,
        'frame_mode'  : 1,  # 0 : continuous  |  1 : uniform intervals
        'num_epochs'  : 20,
        'resume_epoch': 15,   # default : 15
        'batch_size'  : 8,
        'save_freq'   : 5,
        'useTest'     : True,
        'lr'          : 3.5e-4,    # TODO
    }

    model_engine = P3D_Train(params_dict)
    model_engine.model_train()
