#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2019/04/02
author: lujie
"""

# import torch
# from utils.p3d_zoo import *
from IPython import embed

class Test(object):
    kobe = '24'
    def __init__(self):
        self.kobee = '234'


if __name__ == '__main__':

    # model = P3D199(pretrained=False, num_classes=101)
    # model = model.cuda()
    # data = torch.autograd.Variable(torch.rand(10,3,16,160,160)).cuda()   # if modality=='Flow', please change the 2nd dimension 3==>2
    # out = model(data)
    # print(out.size(),out)

    test = Test()
    print(test.kobe, test.kobee)
    test.kobee = '1234'
    print(test.kobe, test.kobee)
