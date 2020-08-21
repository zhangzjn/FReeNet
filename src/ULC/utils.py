import os
import sys
import time
import math

import torch
import torch.nn as nn
import torch.nn.init as init
import random


def adjust_learning_rate(optimizer, lr, epoch, every=100):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = lr * (0.1 ** (epoch // every))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def weight_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.xavier_normal_(m.weight.data, gain=1)
        if m.bias is not None:
            init.constant_(m.bias.data, 0.0)
        # print('Conv')
    elif classname.find('Linear') != -1:
        init.xavier_normal_(m.weight.data, gain=1)
        if m.bias is not None:
            init.constant_(m.bias.data, 0.0)
        # print('Linear')
    elif classname.find('FusePool_zjn') != -1:
        init.constant_(m.weight.data, 1.0)
        # print('FusePool_zjn')
