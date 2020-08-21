from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter
import torchvision.utils as vutils
from data_ULC import *
from ULC_trainer import *
from utils import *
# from .data_L import *
# from .LSNet import *
# from .utils import *

import os
import argparse
import sys
import cv2
import numpy as np

parser = argparse.ArgumentParser(description='FReeNet')
parser.add_argument('--isTrain', default=True, type=bool, help='isTrain')
parser.add_argument('--lr', default=0.0003, type=float, help='learning rate')
parser.add_argument('--gpus', default='0', type=str, help='gpus')
parser.add_argument('--checkpoints', default='checkpoints', type=str, help='checkpoint path')
parser.add_argument('--img_size', default=512, type=int, help='image size')

# RaFD
parser.add_argument('--data', default='RaFD', type=str, choices=['RaFD', 'Multi-PIE'], help='RaFD | Multi-PIE')
parser.add_argument('--name', default='RaFD', type=str, help='checkpoint path')
parser.add_argument('--save_every', default=10, type=int, help='save the model every # epochs')
parser.add_argument('--every', default=60, type=int, help='learning rate decay')
parser.add_argument('--epochs', default=200, type=int, help='epochs')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--resume_epoch', default=None, type=int, help='resume from checkpoint')
# Multi-PIE
# parser.add_argument('--data', default='Multi-PIE', type=str, choices=['RaFD', 'Multi-PIE'], help='RaFD | Multi-PIE')
# parser.add_argument('--name', default='Multi-PIE', type=str, help='checkpoint path')
# parser.add_argument('--save_every', default=100, type=int, help='save the model every # epochs')
# parser.add_argument('--every', default=600, type=int, help='learning rate decay')
# parser.add_argument('--epochs', default=2000, type=int, help='epochs')
# parser.add_argument('--resume', '-r', default=False, type=bool, help='resume from checkpoint')
# parser.add_argument('--resume_epoch', default=None, type=int, help='resume from checkpoint')

opt = parser.parse_args()
opt.gpus = [int(dev) for dev in opt.gpus.split(',')]
torch.cuda.set_device(opt.gpus[0])

for key, val in vars(opt).items():
    if isinstance(val, list):
        val = [str(v) for v in val]
        val = ','.join(val)
    if val is None:
        val = 'None'
    print('{:>20} : {:<50}'.format(key, val))

# Data
print('==> Preparing data..')
if opt.data == 'RaFD':
    trainset = Landmarks_RaFD()
    testset = Landmarks_RaFD()
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=16, shuffle=True, num_workers=1)
    testloader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=False, num_workers=1)
elif opt.data == 'Multi-PIE':
    trainset = Landmarks_PIE()
    testset = Landmarks_PIE()
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=16, shuffle=True, num_workers=1)
    testloader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=False, num_workers=1)
# Model
print('==> Building model..')
net = ULC_trainer(opt)

# Training
def train(epoch=0):
    print('\nEpoch: %d' % epoch)
    net.train()
    net.run(trainloader)


def eval(epoch=0):
    print('\nEpoch: %d' % epoch)
    net.eval()
    net.run(testloader)

# print_network(net)
for epoch in range(1, opt.epochs+1):
    train(epoch)
eval()
