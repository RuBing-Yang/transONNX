#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import os
import time
import torch
from torch import nn, optim
import torch.nn.functional as F
import d2lzh_pytorch as d2l
from d2lzh_pytorch import Residual
import sys
import numpy as np
from skimage import io

sys.path.append("..")
global device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

batch_size = 256  # 256

def resnet_block(in_channels, out_channels, num_residuals, first_block=False):
    if first_block:
        # 第一个模块的通道数同输入通道数一致
        assert in_channels == out_channels
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual(in_channels, out_channels, use_1x1conv=True, stride=2))
        else:
            blk.append(Residual(out_channels, out_channels))
    return nn.Sequential(*blk)

def train_fashion_mnist_Resnet():
    # Resnet-18模型

    net = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
    # 残差块
    net.add_module("resnet_block1", resnet_block(64, 64, 2, first_block=True))
    net.add_module("resnet_block2", resnet_block(64, 128, 2))
    net.add_module("resnet_block3", resnet_block(128, 256, 2))
    net.add_module("resnet_block4", resnet_block(256, 512, 2))
    # 全局平均池化层
    net.add_module("global_avg_pool", d2l.GlobalAvgPool2d())
    # 全连接层输出
    net.add_module("fc", nn.Sequential(d2l.FlattenLayer(), nn.Linear(512, 10)))

    # 训练
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=96)
    lr, num_epochs = 0.001, 5
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    d2l.train_ch5(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs)

    torch.save(net.state_dict(), '_static/model/ResNet.pt')
    torch.save(net, '_static/model/all_ResNet.pt')

    if os.path.exists("_static/img/train0.png"):
        pass
    else:
        save_mnist()

    # 返回训练好的model 和训练data
    return net


def file_to_ResNet():
    if os.path.exists("_static/img/train0.png"):
        pass
    else:
        save_mnist()
    net = torch.load('_static/model/all_ResNet.pt')
    net.eval()
    return net


def save_mnist(train_iter=None):
    # 存10张灰度图，扩展到3通道0~255，尺寸224*224
    n = 0
    if train_iter == None:
        train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=224)
    for train_data in train_iter:
        if n >= 10:
            break
        print(train_data[0].shape)
        img = np.array(train_data[0][n, 0, :, :]).reshape(224, 224, 1) * 255
        img = np.concatenate((img, img, img), axis=2)
        print(img.shape)
        io.imsave("./_static/img/train" + str(n) + ".png", img)
        n += 1