#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Zhu Wenjing
# Date: 2022-03-07
# E-mail: zhuwenjing02@duxiaoman.com

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import RepVGG as rv
import FPN
import PANet
from area_attention import AreaAttention
from g_mlp_pytorch import gMLP
from g_mlp_pytorch import SpatialGatingUnit


## Multi-Scale Block的内容
class MultiConv(nn.Module):
    '''
    Multi-scale block without short-cut connections
    '''
    def __init__(self, channels = 16, **kwargs):
        super(MultiConv, self).__init__()
        ## 不改变原来大小
        self.conv3 = nn.Conv2d(kernel_size=(3, 3), in_channels=channels, out_channels=channels, padding=1)
        self.conv5 = nn.Conv2d(kernel_size=(5, 5), in_channels=channels, out_channels=channels, padding=2)
        self.bn = nn.BatchNorm2d(channels*2)

    def forward(self, x):
        x3 = self.conv3(x)
        x5 = self.conv5(x)
        x = torch.cat((x3,x5),1)
        x = self.bn(x)
        x = F.relu(x)
        return x

# #CSP
# class ResMultiConv(nn.Module):
#     '''
#     Multi-scale block with short-cut connections
#     '''
#     def __init__(self, channels = 16, **kwargs):
#         super(ResMultiConv, self).__init__()
#         self.conv3 = CSPStem(kernel_size=(3, 3), in_channels=channels, out_channels=channels, padding=1)
#         self.conv5 = CSPStem(kernel_size=(5, 5), in_channels=channels, out_channels=channels, padding=2)
#         self.bn = nn.BatchNorm2d(channels*2)
#
#     def forward(self, x):
#         x3 = self.conv3(x) + x
#         x5 = self.conv5(x) + x
#         x = torch.cat((x3,x5),1)
#         x = self.bn(x)
#         x = F.relu(x)
#         return x

# origin
class ResMultiConv(nn.Module):
    '''
    Multi-scale block with short-cut connections
    '''
    def __init__(self, channels = 16, **kwargs):
        super(ResMultiConv, self).__init__()
        self.conv3 = nn.Conv2d(kernel_size=(3, 3), in_channels=channels, out_channels=channels, padding=1)
        self.conv5 = nn.Conv2d(kernel_size=(5, 5), in_channels=channels, out_channels=channels, padding=2)
        self.bn = nn.BatchNorm2d(channels*2)

    def forward(self, x):
        x3 = self.conv3(x) + x
        x5 = self.conv5(x) + x
        x = torch.cat((x3,x5),1)
        x = self.bn(x)
        x = F.relu(x)
        return x

# #MBnet
# class ResMultiConv(nn.Module):
#     '''
#     Multi-scale block with short-cut connections
#     '''
#     def __init__(self, channels = 16, **kwargs):
#         super(ResMultiConv, self).__init__()
#         self.conv3 = MobileNetv2(kernel_size=(3, 3), in_channels=channels, out_channels=channels, padding=1, stride=1)
#         self.conv5 = MobileNetv2(kernel_size=(5, 5), in_channels=channels, out_channels=channels, padding=2, stride=1)
#         self.bn = nn.BatchNorm2d(channels*2)
#
#     def forward(self, x):
#         x3 = self.conv3(x) + x
#         x5 = self.conv5(x) + x
#         x = torch.cat((x3,x5),1)
#         x = self.bn(x)
#         x = F.relu(x)
#         return x

class ResConv3(nn.Module):
    '''
    Resnet with 3x3 kernels
    '''
    def __init__(self, channels = 16, **kwargs):
        super(ResConv3, self).__init__()
        self.conv3 = nn.Conv2d(kernel_size=(3, 3), in_channels=channels, out_channels=2*channels, padding=1)
        self.bn = nn.BatchNorm2d(channels*2)

    def forward(self, x):        
        x = self.conv3(x) + torch.cat((x,x),1)
        x = self.bn(x)
        x = F.relu(x)
        return x    

class ResConv5(nn.Module):
    '''
    Resnet with 5x5 kernels
    '''
    def __init__(self, channels = 16, **kwargs):
        super(ResConv5, self).__init__()
        self.conv5 = nn.Conv2d(kernel_size=(5, 5), in_channels=channels, out_channels=2*channels, padding=2)
        self.bn = nn.BatchNorm2d(channels*2)

    def forward(self, x):        
        x = self.conv5(x) + torch.cat((x,x),1)
        x = self.bn(x)
        x = F.relu(x)
        return x    

class CNN_Area(nn.Module):
    '''Area attention, Mingke Xu, 2020
    '''
    def __init__(self, height=3,width=3,out_size=4, shape=(26,63), **kwargs): # 后面的**kwargs可以接受好多参数存到字典里
        super(CNN_Area, self).__init__()
        self.height=height
        self.width=width
        # self.conv1 = nn.Conv2D(32, (3,3), padding='same', data_format='channels_last',)
        self.conv1a = nn.Conv2d(kernel_size=(10, 2), in_channels=1, out_channels=16, padding=(4, 0))
        self.conv1b = nn.Conv2d(kernel_size=(2, 8), in_channels=1, out_channels=16, padding=(0, 3))
        self.conv2 = nn.Conv2d(kernel_size=(3, 3), in_channels=16, out_channels=32, padding=1)
        self.conv3 = nn.Conv2d(kernel_size=(3, 3), in_channels=32, out_channels=48, padding=1)
        self.conv4 = nn.Conv2d(kernel_size=(3, 3), in_channels=48, out_channels=64, padding=1)
        self.conv5 = nn.Conv2d(kernel_size=(3, 3), in_channels=64, out_channels=80, padding=1)
        # self.conv6 = nn.Conv2D(128, (3, 3), padding= )#
        self.maxp = nn.MaxPool2d(kernel_size=(2, 2))
        self.bn1a = nn.BatchNorm2d(16)
        self.bn1b = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(48)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(80)
        # self.gap = nn.AdaptiveAvgPool2d(1) 把每个batch在channel不变的情况下算个平均值

        i = 80 * ((shape[0]-1)//2) * ((shape[1]-1)//4) 
        self.fc = nn.Linear(in_features=i, out_features=4)

    def forward(self, *input):
        x = input[0]
        xa = self.conv1a(x)
        xa = self.bn1a(xa)
        xa=F.relu(xa)
        xb = self.conv1b(x)
        xb = self.bn1b(xb)
        xb = F.relu(xb)
        x = torch.cat((xa, xb), 2)

        x = self.conv2(x)
        x = self.bn2(x)
        x=F.relu(x)
        x = self.maxp(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.maxp(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.conv5(x)
        x = self.bn5(x)
        x = F.relu(x)

        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        
        return x

class CNN_AttnPooling(nn.Module):
    '''Head atention, Mingke Xu, 2020
    Attention pooling, Pengcheng Li, Interspeech, 2019
    '''
    def __init__(self, head=4, attn_hidden=64, shape=(26,63), **kwargs):
        super(CNN_AttnPooling, self).__init__()
        self.head = head
        self.attn_hidden = attn_hidden
        self.conv1a = nn.Conv2d(kernel_size=(10, 2), in_channels=1, out_channels=8, padding=(4, 0))
        self.conv1b = nn.Conv2d(kernel_size=(2, 8), in_channels=1, out_channels=8, padding=(0, 3))
        self.conv2 = nn.Conv2d(kernel_size=(3, 3), in_channels=16, out_channels=32, padding=1)
        self.conv3 = nn.Conv2d(kernel_size=(3, 3), in_channels=32, out_channels=48, padding=1)
        self.conv4 = nn.Conv2d(kernel_size=(3, 3), in_channels=48, out_channels=64, padding=1)
        self.conv5 = nn.Conv2d(kernel_size=(3, 3), in_channels=64, out_channels=80, padding=1)
        self.maxp = nn.MaxPool2d(kernel_size=(2, 2))
        self.bn1a = nn.BatchNorm2d(8)
        self.bn1b = nn.BatchNorm2d(8)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(48)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(80)
        self.top_down = nn.Conv2d(kernel_size=(1, 1), in_channels=80, out_channels=4)
        self.bottom_up = nn.Conv2d(kernel_size=(1, 1), in_channels=80, out_channels=1)
        # i = 80 * ((shape[0]-1)//4) * ((shape[1]-1)//4) 
        # self.fc = nn.Linear(in_features=i, out_features=4)

    def forward(self, *input):
        xa = self.conv1a(input[0])
        xa = self.bn1a(xa)

        xa = F.relu(xa)
        xb = self.conv1b(input[0])
        xb = self.bn1b(xb)

        xb = F.relu(xb)
        x = torch.cat((xa, xb), 1)
        x = self.conv2(x)
        x = self.bn2(x)

        x = F.relu(x)
        x = self.maxp(x)
        x = self.conv3(x)
        x = self.bn3(x)

        x= F.relu(x)
        x = self.maxp(x)
        x = self.conv4(x)
        x = self.bn4(x)

        x = F.relu(x)

        x = self.conv5(x)
        x = self.bn5(x)

        x = F.relu(x)

        x1 = self.top_down(x)
        x1 = F.softmax(x1,1)
        x2 = self.bottom_up(x)

        x = x1 * x2

        # x = x.sum((2,3))        
        x = x.mean((2,3))        

        return x

## Global Average Pooling常常被用来将特征图降维，同时剔除无用的空间信息
class CNN_GAP(nn.Module):
    '''Head atention, Mingke Xu, 2020
    Attention pooling, Pengcheng Li, Interspeech, 2019
    '''
    def __init__(self, head=4, attn_hidden=64, shape=(26,63), **kwargs):
        super(CNN_GAP, self).__init__()
        self.head = head
        self.attn_hidden = attn_hidden
        self.conv1a = nn.Conv2d(kernel_size=(10, 2), in_channels=1, out_channels=8, padding=(4, 0))
        self.conv1b = nn.Conv2d(kernel_size=(2, 8), in_channels=1, out_channels=8, padding=(0, 3))
        self.conv2 = nn.Conv2d(kernel_size=(3, 3), in_channels=16, out_channels=32, padding=1)
        self.conv3 = nn.Conv2d(kernel_size=(3, 3), in_channels=32, out_channels=48, padding=1)
        self.conv4 = nn.Conv2d(kernel_size=(3, 3), in_channels=48, out_channels=64, padding=1)
        self.conv5 = nn.Conv2d(kernel_size=(3, 3), in_channels=64, out_channels=80, padding=1)
        self.maxp = nn.MaxPool2d(kernel_size=(2, 2))
        self.bn1a = nn.BatchNorm2d(8)
        self.bn1b = nn.BatchNorm2d(8)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(48)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(80)
        self.gap = nn.AdaptiveAvgPool2d(1)
        i = 80 * ((shape[0]-1)//4) * ((shape[1]-1)//4) 
        self.fc = nn.Linear(in_features=i, out_features=4)

    def forward(self, *input):
        xa = self.conv1a(input[0])
        xa = self.bn1a(xa)

        xa = F.relu(xa)
        xb = self.conv1b(input[0])
        xb = self.bn1b(xb)

        xb = F.relu(xb)
        x = torch.cat((xa, xb), 1)
        x = self.conv2(x)
        x = self.bn2(x)

        x = F.relu(x)
        x = self.maxp(x)
        x = self.conv3(x)
        x = self.bn3(x)

        x= F.relu(x)
        x = self.maxp(x)
        x = self.conv4(x)
        x = self.bn4(x)

        x = F.relu(x)

        x = self.conv5(x)
        x = self.bn5(x)

        x = F.relu(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        return x

## 多头注意力机制
class MHCNN(nn.Module):
    '''
    Multi-Head Attention
    '''
    def __init__(self, head=4, attn_hidden=64,**kwargs):
        super(MHCNN, self).__init__()
        self.head = head
        self.attn_hidden = attn_hidden
        self.conv1a = nn.Conv2d(kernel_size=(10, 2), in_channels=1, out_channels=8, padding=(4, 0))
        self.conv1b = nn.Conv2d(kernel_size=(2, 8), in_channels=1, out_channels=8, padding=(0, 3))
        self.conv2 = nn.Conv2d(kernel_size=(3, 3), in_channels=16, out_channels=32, padding=1)
        self.conv3 = nn.Conv2d(kernel_size=(3, 3), in_channels=32, out_channels=48, padding=1)
        self.conv4 = nn.Conv2d(kernel_size=(3, 3), in_channels=48, out_channels=64, padding=1)
        self.conv5 = nn.Conv2d(kernel_size=(3, 3), in_channels=64, out_channels=80, padding=1)
        self.maxp = nn.MaxPool2d(kernel_size=(2, 2))
        self.bn1a = nn.BatchNorm2d(8)
        self.bn1b = nn.BatchNorm2d(8)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(48)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(80)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(in_features=self.attn_hidden, out_features=4)
        self.dropout = nn.Dropout(0.5) # 每一轮训练过程中，大约有50%的神经元会在指定的层中被"退出"或关闭。
        self.attention_query = nn.ModuleList() # 简单的容器，可以容纳任意数量的 nn.Module
        self.attention_key = nn.ModuleList()
        self.attention_value = nn.ModuleList()

        for i in range(self.head): # 想创建多少组自注意力参数
            self.attention_query.append(nn.Conv2d(in_channels=80, out_channels=self.attn_hidden, kernel_size=1))
            self.attention_key.append(nn.Conv2d(in_channels=80, out_channels=self.attn_hidden, kernel_size=1))
            self.attention_value.append(nn.Conv2d(in_channels=80, out_channels=self.attn_hidden, kernel_size=1))

    def forward(self, *input):
        xa = self.conv1a(input[0])
        xa = self.bn1a(xa)

        xa = F.relu(xa)
        xb = self.conv1b(input[0])
        xb = self.bn1b(xb)

        xb = F.relu(xb)
        x = torch.cat((xa, xb), 1)
        x = self.conv2(x)
        x = self.bn2(x)

        x = F.relu(x)
        x = self.maxp(x)
        x = self.conv3(x)
        x = self.bn3(x)

        x= F.relu(x)
        x = self.maxp(x)
        x = self.conv4(x)
        x = self.bn4(x)

        x = F.relu(x)

        x = self.conv5(x)
        x = self.bn5(x)

        x = F.relu(x)
        # #attention

        attn = None
        for i in range(self.head):
            Q = self.attention_query[i](x)
            K = self.attention_key[i](x)
            V = self.attention_value[i](x)
            attention = F.softmax(torch.mul(Q, K),dim=1)
            attention = torch.mul(attention, V)

            if (attn is None):
                attn = attention
            else:
                attn = torch.cat((attn, attention), 2)
        x = attn
        x = F.relu(x)
        x = self.gap(x)

        x = x.reshape(x.shape[0], x.shape[1] * x.shape[2] * x.shape[3])

        x = self.fc(x)
        return x

class MHCNN_AreaConcat(nn.Module):
    def __init__(self, head=4, attn_hidden=64, shape=(26,63), **kwargs):
        super(MHCNN_AreaConcat, self).__init__()
        self.head = head
        self.attn_hidden = attn_hidden
        self.conv1a = nn.Conv2d(kernel_size=(10, 2), in_channels=1, out_channels=16, padding=(4, 0))
        self.conv1b = nn.Conv2d(kernel_size=(2, 8), in_channels=1, out_channels=16, padding=(0, 3))
        self.conv2 = nn.Conv2d(kernel_size=(3, 3), in_channels=16, out_channels=32, padding=1)
        self.conv3 = nn.Conv2d(kernel_size=(3, 3), in_channels=32, out_channels=48, padding=1)
        self.conv4 = nn.Conv2d(kernel_size=(3, 3), in_channels=48, out_channels=64, padding=1)
        self.conv5 = nn.Conv2d(kernel_size=(3, 3), in_channels=64, out_channels=80, padding=1)
        self.maxp = nn.MaxPool2d(kernel_size=(2, 2))
        self.bn1a = nn.BatchNorm2d(16)
        self.bn1b = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(48)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(80)
        self.gap = nn.AdaptiveAvgPool2d(1)
        i = self.attn_hidden * ((shape[0]-1)//2) * ((shape[1]-1)//4) * self.head
        self.fc = nn.Linear(in_features=i, out_features=4)
        self.dropout = nn.Dropout(0.5)
        self.attention_query = nn.ModuleList()
        self.attention_key = nn.ModuleList()
        self.attention_value = nn.ModuleList()

        for i in range(self.head):
            self.attention_query.append(nn.Conv2d(in_channels=80, out_channels=self.attn_hidden, kernel_size=1))
            self.attention_key.append(nn.Conv2d(in_channels=80, out_channels=self.attn_hidden, kernel_size=1))
            self.attention_value.append(nn.Conv2d(in_channels=80, out_channels=self.attn_hidden, kernel_size=1))

    def forward(self, *input):
        xa = self.conv1a(input[0])
        xa = self.bn1a(xa)

        xa = F.relu(xa)
        xb = self.conv1b(input[0])
        xb = self.bn1b(xb)

        xb = F.relu(xb)
        x = torch.cat((xa, xb), 2)
        x = self.conv2(x)
        x = self.bn2(x)

        x = F.relu(x)
        x = self.maxp(x)
        x = self.conv3(x)
        x = self.bn3(x)

        x= F.relu(x)
        x = self.maxp(x)
        x = self.conv4(x)
        x = self.bn4(x)

        x = F.relu(x)

        x = self.conv5(x)
        x = self.bn5(x)

        x = F.relu(x)
        # #attention

        attn = None
        for i in range(self.head):
            Q = self.attention_query[i](x)
            K = self.attention_key[i](x)
            V = self.attention_value[i](x)
            attention = F.softmax(torch.mul(Q, K),dim=1)
            attention = torch.mul(attention, V)

            if (attn is None):
                attn = attention
            else:
                attn = torch.cat((attn, attention), 2)
        x = attn
        x = F.relu(x)
        # x = self.gap(x)

        x = x.reshape(x.shape[0], x.shape[1] * x.shape[2] * x.shape[3])

        x = self.fc(x)
        return x

class MHCNN_AreaConcat_gap(nn.Module):
    def __init__(self, head=4, attn_hidden=64,**kwargs):
        super(MHCNN_AreaConcat_gap, self).__init__()
        self.head = head
        self.attn_hidden = attn_hidden
        self.conv1a = nn.Conv2d(kernel_size=(10, 2), in_channels=1, out_channels=16, padding=(4, 0))
        self.conv1b = nn.Conv2d(kernel_size=(2, 8), in_channels=1, out_channels=16, padding=(0, 3))
        self.conv2 = nn.Conv2d(kernel_size=(3, 3), in_channels=16, out_channels=32, padding=1)
        self.conv3 = nn.Conv2d(kernel_size=(3, 3), in_channels=32, out_channels=48, padding=1)
        self.conv4 = nn.Conv2d(kernel_size=(3, 3), in_channels=48, out_channels=64, padding=1)
        self.conv5 = nn.Conv2d(kernel_size=(3, 3), in_channels=64, out_channels=80, padding=1)
        self.maxp = nn.MaxPool2d(kernel_size=(2, 2))
        self.bn1a = nn.BatchNorm2d(16)
        self.bn1b = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(48)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(80)
        self.gap = nn.AdaptiveAvgPool2d(1)
        i = self.attn_hidden 
        self.fc = nn.Linear(in_features=i, out_features=4)
        self.dropout = nn.Dropout(0.5)
        self.attention_query = nn.ModuleList()
        self.attention_key = nn.ModuleList()
        self.attention_value = nn.ModuleList()

        for i in range(self.head):
            self.attention_query.append(nn.Conv2d(in_channels=80, out_channels=self.attn_hidden, kernel_size=1))
            self.attention_key.append(nn.Conv2d(in_channels=80, out_channels=self.attn_hidden, kernel_size=1))
            self.attention_value.append(nn.Conv2d(in_channels=80, out_channels=self.attn_hidden, kernel_size=1))

    def forward(self, *input):
        xa = self.conv1a(input[0])
        xa = self.bn1a(xa)

        xa = F.relu(xa)
        xb = self.conv1b(input[0])
        xb = self.bn1b(xb)

        xb = F.relu(xb)
        x = torch.cat((xa, xb), 2)
        x = self.conv2(x)
        x = self.bn2(x)

        x = F.relu(x)
        x = self.maxp(x)
        x = self.conv3(x)
        x = self.bn3(x)

        x= F.relu(x)
        x = self.maxp(x)
        x = self.conv4(x)
        x = self.bn4(x)

        x = F.relu(x)

        x = self.conv5(x)
        x = self.bn5(x)

        x = F.relu(x)
        # #attention

        attn = None
        for i in range(self.head):
            Q = self.attention_query[i](x)
            K = self.attention_key[i](x)
            V = self.attention_value[i](x)
            attention = F.softmax(torch.mul(Q, K),dim=1)
            attention = torch.mul(attention, V)

            if (attn is None):
                attn = attention
            else:
                attn = torch.cat((attn, attention), 2)
        x = attn
        x = F.relu(x)
        x = self.gap(x)

        x = x.reshape(x.shape[0], x.shape[1] * x.shape[2] * x.shape[3])

        x = self.fc(x)
        return x

class MHCNN_AreaConcat_gap1(nn.Module):
    def __init__(self, head=4, attn_hidden=64, shape=(26,63), **kwargs):
        super(MHCNN_AreaConcat_gap1, self).__init__()
        self.head = head
        self.attn_hidden = 32
        self.conv1a = nn.Conv2d(kernel_size=(10, 2), in_channels=1, out_channels=16, padding=(4, 0))
        self.conv1b = nn.Conv2d(kernel_size=(2, 8), in_channels=1, out_channels=16, padding=(0, 3))
        self.conv2 = nn.Conv2d(kernel_size=(3, 3), in_channels=16, out_channels=32, padding=1)
        self.conv3 = nn.Conv2d(kernel_size=(3, 3), in_channels=32, out_channels=48, padding=1)
        self.conv4 = nn.Conv2d(kernel_size=(3, 3), in_channels=48, out_channels=64, padding=1)
        self.conv5 = nn.Conv2d(kernel_size=(3, 3), in_channels=64, out_channels=80, padding=1)
        self.maxp = nn.MaxPool2d(kernel_size=(2, 2))
        self.bn1a = nn.BatchNorm2d(16)
        self.bn1b = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(48)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(80)
        self.gap = nn.AdaptiveAvgPool2d(1)
        i = self.attn_hidden 
        self.fc = nn.Linear(in_features=(shape[0]-1)//2, out_features=4)
        self.dropout = nn.Dropout(0.5)
        self.attention_query = nn.ModuleList()
        self.attention_key = nn.ModuleList()
        self.attention_value = nn.ModuleList()

        for i in range(self.head):
            self.attention_query.append(nn.Conv2d(in_channels=80, out_channels=self.attn_hidden, kernel_size=1))
            self.attention_key.append(nn.Conv2d(in_channels=80, out_channels=self.attn_hidden, kernel_size=1))
            self.attention_value.append(nn.Conv2d(in_channels=80, out_channels=self.attn_hidden, kernel_size=1))

    def forward(self, *input):
        xa = self.conv1a(input[0])
        xa = self.bn1a(xa)

        xa = F.relu(xa)
        xb = self.conv1b(input[0])
        xb = self.bn1b(xb)

        xb = F.relu(xb)
        x = torch.cat((xa, xb), 2)
        x = self.conv2(x)
        x = self.bn2(x)

        x = F.relu(x)
        x = self.maxp(x)
        x = self.conv3(x)
        x = self.bn3(x)

        x= F.relu(x)
        x = self.maxp(x)
        x = self.conv4(x)
        x = self.bn4(x)

        x = F.relu(x)

        x = self.conv5(x)
        x = self.bn5(x)

        x = F.relu(x)
        # #attention

        attn = None
        for i in range(self.head):
            Q = self.attention_query[i](x)
            K = self.attention_key[i](x)
            V = self.attention_value[i](x)
            attention = F.softmax(torch.mul(Q, K),dim=1)
            attention = torch.mul(attention, V)

            if (attn is None):
                attn = attention
            else:
                attn = torch.cat((attn, attention), 3)
        x = attn
        x = x.contiguous().permute(0, 2, 3, 1)
        x = F.relu(x)
        x = self.gap(x)

        x = x.reshape(x.shape[0], -1)

        x = self.fc(x)
        return x

class AACNN(nn.Module):
    '''
    Area Attention, ICASSP 2020
    '''
    def __init__(self, height=3,width=3,out_size=4, shape=(26,63), **kwargs):
        super(AACNN, self).__init__()
        self.height=height
        self.width=width
        # self.conv1 = nn.Conv2D(32, (3,3), padding='same', data_format='channels_last',)
        self.conv1a = nn.Conv2d(kernel_size=(10, 2), in_channels=1, out_channels=16, padding=(4, 0))
        self.conv1b = nn.Conv2d(kernel_size=(2, 8), in_channels=1, out_channels=16, padding=(0, 3))
        self.conv2 = nn.Conv2d(kernel_size=(3, 3), in_channels=16, out_channels=32, padding=1)
        self.conv3 = nn.Conv2d(kernel_size=(3, 3), in_channels=32, out_channels=48, padding=1)
        self.conv4 = nn.Conv2d(kernel_size=(3, 3), in_channels=48, out_channels=64, padding=1)
        self.conv5 = nn.Conv2d(kernel_size=(3, 3), in_channels=64, out_channels=80, padding=1)
        # self.conv6 = nn.Conv2D(128, (3, 3), padding= )#
        self.maxp = nn.MaxPool2d(kernel_size=(2, 2))
        self.bn1a = nn.BatchNorm2d(16)
        self.bn1b = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(48)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(80)
        # self.gap = nn.AdaptiveAvgPool2d(1)

        i = 80 * ((shape[0] - 1)//2) * ((shape[1] - 1)//4)
        self.fc = nn.Linear(in_features=i, out_features=4)
        # self.dropout = nn.Dropout(0.5)

        self.area_attention = AreaAttention(
            key_query_size=80,
            area_key_mode='mean',
            area_value_mode='sum',
            max_area_height=height,
            max_area_width=width,
            dropout_rate=0.5,
        )


    def forward(self, *input):
        x = input[0]
        xa = self.conv1a(x)
        xa = self.bn1a(xa)
        xa=F.relu(xa)
        xb = self.conv1b(x)
        xb = self.bn1b(xb)
        xb = F.relu(xb)
        x = torch.cat((xa, xb), 2)

        x = self.conv2(x)
        x = self.bn2(x)
        x=F.relu(x)
        x = self.maxp(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.maxp(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.conv5(x)
        x = self.bn5(x)
        x = F.relu(x)

        shape = x.shape
        x = x.contiguous().permute(0, 2, 3, 1).view(shape[0], shape[3]*shape[2], shape[1])
        x = self.area_attention(x,x,x)
        x = F.relu(x)
        x = x.reshape(*shape)

        x = x.reshape(x.shape[0], -1)

        x = self.fc(x)
        
        return x

class AACNN_HeadConcat(nn.Module):
    def __init__(self, height=3,width=3,out_size=4, shape=(26,63), **kwargs):
        super(AACNN_HeadConcat, self).__init__()
        self.height=height
        self.width=width
        self.conv1a = nn.Conv2d(kernel_size=(10, 2), in_channels=1, out_channels=8, padding=(4, 0))
        self.conv1b = nn.Conv2d(kernel_size=(2, 8), in_channels=1, out_channels=8, padding=(0, 3))
        self.conv2 = nn.Conv2d(kernel_size=(3, 3), in_channels=16, out_channels=32, padding=1)
        self.conv3 = nn.Conv2d(kernel_size=(3, 3), in_channels=32, out_channels=48, padding=1)
        self.conv4 = nn.Conv2d(kernel_size=(3, 3), in_channels=48, out_channels=64, padding=1)
        self.conv5 = nn.Conv2d(kernel_size=(3, 3), in_channels=64, out_channels=80, padding=1)
        self.maxp = nn.MaxPool2d(kernel_size=(2, 2))
        self.bn1a = nn.BatchNorm2d(8)
        self.bn1b = nn.BatchNorm2d(8)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(48)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(80)
        i = 80 * ((shape[0] - 1)//4) * ((shape[1] - 1)//4)
        self.fc = nn.Linear(in_features=i, out_features=4)
        self.dropout = nn.Dropout(0.5)
        self.area_attention = AreaAttention(
            key_query_size=80,
            area_key_mode='mean',
            area_value_mode='sum',
            max_area_height=height,
            max_area_width= width,
            dropout_rate=0.5,
            # top_k_areas=0
        )


    def forward(self, *input):
        x = input[0]
        xa = self.conv1a(x)
        xa = self.bn1a(xa)
        xa=F.relu(xa)
        xb = self.conv1b(x)
        xb = self.bn1b(xb)
        xb = F.relu(xb)
        x = torch.cat((xa, xb), 1)

        x = self.conv2(x)
        x = self.bn2(x)
        x=F.relu(x)
        x = self.maxp(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.maxp(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.conv5(x)
        x = self.bn5(x)
        x = F.relu(x)

        # flatten
        shape = x.shape
        x = x.contiguous().permute(0, 2, 3, 1).view(shape[0], shape[3]*shape[2], shape[1])
        
        x = self.area_attention(x,x,x)
        x = F.relu(x)

        x = x.reshape(x.shape[0], -1)

        x = self.fc(x)
        
        return x

class GLAM5(nn.Module):
    '''
    GLobal-Aware Multiscale block with 5x5 convolutional kernels in CNN architecture
    '''
    def __init__(self, shape=(26,63), **kwargs):
        super(GLAM5, self).__init__()
        self.conv1a = nn.Conv2d(kernel_size=(5, 1), in_channels=1, out_channels=16, padding=(2, 0))
        self.conv1b = nn.Conv2d(kernel_size=(1, 5), in_channels=1, out_channels=16, padding=(0, 2))
        self.conv2 = ResMultiConv(16)
        self.conv3 = ResMultiConv(32)
        self.conv4 = ResMultiConv(64)
        self.conv5 = nn.Conv2d(kernel_size=(5, 5), in_channels=128, out_channels=128, padding=2)
        self.maxp = nn.MaxPool2d(kernel_size=(2, 2))
        self.bn1a = nn.BatchNorm2d(16)
        self.bn1b = nn.BatchNorm2d(16)
        self.bn5 = nn.BatchNorm2d(128)
        dim = (shape[0]//2) * (shape[1]//4)
        i = 128 * dim
        self.fc = nn.Linear(in_features=i, out_features=4)
        self.dropout = nn.Dropout(0.5)

        self.gmlp = gMLP(dim = dim, depth = 1, seq_len = 128, act = nn.Tanh())

    def forward(self, *input):
        # input[0]: torch.Size([32, 1, 26, 63])
        xa = self.conv1a(input[0]) # (32, 16, 25, 62)
        xa = self.bn1a(xa) # (32, 16, 25, 62)

        xa = F.relu(xa)
        xb = self.conv1b(input[0])
        xb = self.bn1b(xb)

        xb = F.relu(xb)
        x = torch.cat((xa, xb), 2) # (32, 16, 50, 62)

        x = self.conv2(x) # (32, 32, 50, 62)
        x = self.maxp(x)
        x = self.conv3(x) # (32, 64, 25, 31)
        x = self.maxp(x)
        x = self.conv4(x) # (32, 128, 12, 15)

        x = self.conv5(x)  # (32, 128, 12, 15)
        x = self.bn5(x)
        x = F.relu(x)

        # flatten
        shape = x.shape
        x = x.view(*x.shape[:-2],-1)

        x = self.gmlp(x)
        x = F.relu(x)

        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        return x

class Mish(nn.Module):
    def __init__(self):
        super(Mish, self).__init__()

    def forward(self, x):
        return x * torch.tanh(F.softplus(x))

class BN_Conv_Mish(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation=1, groups=1, bias=False):
        super(BN_Conv_Mish, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation=dilation,
                              groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        return Mish()(out)

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = BN_Conv_Mish(channels, channels, 1, 1, 0)     # always use samepadding
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1, bias=False)
        self.bn = nn.BatchNorm2d(channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.bn(out) + x
        return Mish()(out)


class CSPStem(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, stride = 1):
        super(CSPStem, self).__init__()
        self.dsample = BN_Conv_Mish(in_channels, out_channels, 3, 1, 1)
        self.trans_0 = BN_Conv_Mish(out_channels, out_channels // 2, 1, 1, 0)
        self.trans_1 = BN_Conv_Mish(out_channels, out_channels // 2, 1, 1, 0)
        self.blocks = ResidualBlock(out_channels // 2)
        self.trans_cat = BN_Conv_Mish(out_channels, out_channels, 1, 1, 0)

    def forward(self, x):
        x = self.dsample(x)
        out_0 = self.trans_0(x)
        out_1 = self.trans_1(x)
        out_1 = self.blocks(out_1)
        out = torch.cat((out_0, out_1), 1)
        out = self.trans_cat(out)
        return out

class MobileNetv2(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, shortcut = 0):
        super(MobileNetv2, self).__init__()
        self.use_shortcut = shortcut
        self.lift_ch = nn.Conv2d(in_channels = in_channels, out_channels = 6 * in_channels, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(6 * in_channels)
        self.dw = nn.Conv2d(in_channels * 6, in_channels * 6, kernel_size=3, padding=1, groups=in_channels)
        self.bn2 = nn.BatchNorm2d(in_channels * 6)
        self.proj = nn.Conv2d(in_channels * 6, out_channels, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(out_channels)

    def forward(self,  x):
        residual = x
        x = self.lift_ch(x)
        x = self.bn1(x)
        x = F.relu6(x, inplace=False)
        x = self.dw(x)
        x = self.bn2(x)
        x = F.relu6(x, inplace=False)
        x = self.proj(x)
        x = self.bn3(x)
        if self.use_shortcut:
            x += residual
        return x


# # CSP
# class GLAM(nn.Module):
#     def __init__(self, shape = (26,63), **kwargs):
#         super(GLAM, self).__init__()
#         self.conva1 = CSPStem(in_channels=1, out_channels=16, kernel_size=(1, 3), stride=1, padding=(0, 1))
#         self.batchnormb1 = nn.BatchNorm2d(16)
#         self.conva2 = CSPStem(in_channels=1, out_channels=16, kernel_size=(3, 1), stride=1, padding=(1, 0))
#         self.batchnormb2 = nn.BatchNorm2d(16)
#         self.MSB1 = ResMultiConv(16)
#         self.maxp1 = nn.MaxPool2d(kernel_size=(2, 2))
#         self.MSB2 = ResMultiConv(32)
#         self.maxp2 = nn.MaxPool2d(kernel_size=(2, 2))
#         self.MSB3 = ResMultiConv(64)
#         self.conv3 = CSPStem(in_channels=128, out_channels=128, kernel_size=(5, 5), stride=1, padding=2)
#         self.bn = nn.BatchNorm2d(128)
#         dim = (shape[0]//2) * (shape[1]//4)
#         self.gmlp = gMLP(dim=dim, depth=1, seq_len=128, act=nn.Tanh())
#         self.fc = nn.Linear(in_features=dim*128, out_features=4)
#
#     def forward(self, *input):
#         # 输入数据的大小：[32, 1, 26, 57]
#         xa = self.conva1(input[0])
#         xa = self.batchnormb1(xa)
#
#         xb = self.conva2(input[0])
#         xb = self.batchnormb2(xb)
#
#         xa = F.relu(xa) # [32, 16, 26, 57]
#         xb = F.relu(xb) # [32, 16, 26, 57]
#
#         x = torch.cat([xa, xb], dim = 2) # [32, 16, 52, 57]
#         x = self.MSB1(x)
#         x = self.maxp1(x)
#         x = self.MSB2(x)
#         x = self.maxp2(x)
#         x = self.MSB3(x) # [32, 128, 13, 14]
#         x = self.conv3(x)
#         x = self.bn(x)
#
#         x = F.relu(x)
#         x = x.view(*x.shape[:-2], -1)
#
#         x = self.gmlp(x)
#         x = x.reshape(x.shape[0], -1)
#         x = self.fc(x)
#         return x
#
#     class GLAM(nn.Module):
#         def __init__(self, shape=(26, 63), **kwargs):
#             super(GLAM, self).__init__()
#             self.conva1 = CSPStem(in_channels=1, out_channels=16, kernel_size=(1, 3), stride=1, padding=(0, 1))
#             self.batchnormb1 = nn.BatchNorm2d(16)
#             self.conva2 = CSPStem(in_channels=1, out_channels=16, kernel_size=(3, 1), stride=1, padding=(1, 0))
#             self.batchnormb2 = nn.BatchNorm2d(16)
#             self.MSB1 = ResMultiConv(16)
#             self.maxp1 = nn.MaxPool2d(kernel_size=(2, 2))
#             self.MSB2 = ResMultiConv(32)
#             self.maxp2 = nn.MaxPool2d(kernel_size=(2, 2))
#             self.MSB3 = ResMultiConv(64)
#             self.conv3 = CSPStem(in_channels=128, out_channels=128, kernel_size=(5, 5), stride=1, padding=2)
#             self.bn = nn.BatchNorm2d(128)
#             dim = (shape[0] // 2) * (shape[1] // 4)
#             self.gmlp = gMLP(dim=dim, depth=1, seq_len=128, act=nn.Tanh())
#             self.fc = nn.Linear(in_features=dim * 128, out_features=4)
#
#         def forward(self, *input):
#             # 输入数据的大小：[32, 1, 26, 57]
#             xa = self.conva1(input[0])
#             xa = self.batchnormb1(xa)
#
#             xb = self.conva2(input[0])
#             xb = self.batchnormb2(xb)
#
#             xa = F.relu(xa)  # [32, 16, 26, 57]
#             xb = F.relu(xb)  # [32, 16, 26, 57]
#
#             x = torch.cat([xa, xb], dim=2)  # [32, 16, 52, 57]
#             x = self.MSB1(x)
#             x = self.maxp1(x)
#             x = self.MSB2(x)
#             x = self.maxp2(x)
#             x = self.MSB3(x)  # [32, 128, 13, 14]
#             x = self.conv3(x)
#             x = self.bn(x)
#
#             x = F.relu(x)
#             x = x.view(*x.shape[:-2], -1)
#
#             x = self.gmlp(x)
#             x = x.reshape(x.shape[0], -1)
#             x = self.fc(x)
#             return x

# # final
# class GLAM(nn.Module):
#     def __init__(self, shape = (26,63), **kwargs):
#         super(GLAM, self).__init__()
#         self.convy = nn.Conv2d(in_channels=16, out_channels=256,kernel_size=(5, 5), stride=8, padding=(0, 0))
#         self.convextra = ResMultiConv(128)
#         self.conva1 = CSPStem(in_channels=1, out_channels=16, kernel_size=(1, 3), stride=1, padding=(0, 1))
#         self.batchnormb1 = nn.BatchNorm2d(16)
#         self.conva2 = CSPStem(in_channels=1, out_channels=16, kernel_size=(3, 1), stride=1, padding=(1, 0))
#         self.batchnormb2 = nn.BatchNorm2d(16)
#         self.MSB1 = ResMultiConv(16)
#         self.maxp1 = nn.MaxPool2d(kernel_size=(2, 2))
#         self.MSB2 = ResMultiConv(32)
#         self.maxp2 = nn.MaxPool2d(kernel_size=(2, 2))
#         self.MSB3 = ResMultiConv(64)
#         self.maxp3 = nn.MaxPool2d(kernel_size=(2, 2))
#         self.conv3 = CSPStem(in_channels=256, out_channels=256, kernel_size=(5, 5), stride=1, padding=2)
#         self.bn = nn.BatchNorm2d(256)
#         dim = (shape[0]//4) * (shape[1]//8)
#         self.gmlp = gMLP(dim=dim, depth=1, seq_len=256, act=nn.Tanh())
#         self.fc = nn.Linear(in_features=dim*256, out_features=4)
#
#     def forward(self, *input):
#         # 输入数据的大小：[32, 1, 26, 57]
#         xa = self.conva1(input[0])
#         xa = self.batchnormb1(xa)
#
#         xb = self.conva2(input[0])
#         xb = self.batchnormb2(xb)
#
#         xa = F.relu(xa) # [32, 16, 26, 57]
#         xb = F.relu(xb) # [32, 16, 26, 57]
#
#         x = torch.cat([xa, xb], dim = 2) # [32, 16, 52, 57]
#         y = self.convy(x)
#         x = self.MSB1(x)
#         x = self.maxp1(x)
#         x = self.MSB2(x)
#         x = self.maxp2(x)
#         x = self.MSB3(x) # [32, 128, 13, 14]
#         x = self.maxp3(x)
#         x = self.convextra(x)
#         x = self.conv3(x)
#         x = x + y
#         x = self.bn(x)
#
#         x = F.relu(x)
#         x = x.view(*x.shape[:-2], -1)
#
#         x = self.gmlp(x)
#         x = x.reshape(x.shape[0], -1)
#         x = self.fc(x)
#         return x

# # conv
# class GLAM(nn.Module):
#     '''
#     GLobal-Aware Multiscale block with 3x3 convolutional kernels in CNN architecture
#     '''
#     def __init__(self, shape=(26,63), **kwargs):
#         super(GLAM, self).__init__()
#         self.conv1a = nn.Conv2d(kernel_size=(3, 1), in_channels=1, out_channels=16, padding=(1, 0))
#         self.conv1b = nn.Conv2d(kernel_size=(1, 3), in_channels=1, out_channels=16, padding=(0, 1))
#         self.conv2 = ResMultiConv(16)
#         self.conv3 = ResMultiConv(32)
#         self.conv4 = ResMultiConv(64)
#         self.convextra = ResMultiConv(128)
#         self.conv5 = nn.Conv2d(kernel_size=(5, 5), in_channels=256, out_channels=256, padding=2)
#         self.maxp = nn.MaxPool2d(kernel_size=(2, 2))
#         self.bn1a = nn.BatchNorm2d(16)
#         self.bn1b = nn.BatchNorm2d(16)
#         self.bn5 = nn.BatchNorm2d(256)
#         dim = (shape[0]//4) * (shape[1]//8)
#         i = 256 * dim
#         self.fc = nn.Linear(in_features=i, out_features=4)
#         self.dropout = nn.Dropout(0.5)
#
#         self.gmlp = gMLP(dim = dim, depth = 1, seq_len = 256, act = nn.Tanh())
#
#     def forward(self, *input):
#         # input[0]: torch.Size([32, 1, 26, 57])
#         xa = self.conv1a(input[0]) # (32, 16, 26, 57)
#         xa = self.bn1a(xa) # (32, 16, 26, 57)
#
#         xa = F.relu(xa)
#         xb = self.conv1b(input[0])
#         xb = self.bn1b(xb)
#
#         xb = F.relu(xb)
#         x = torch.cat((xa, xb), 2) # (32, 16, 52, 57)
#
#         x = self.conv2(x) # (32, 32, 52, 57)
#         x = self.maxp(x)
#         x = self.conv3(x) # (32, 64, 26, 28)
#         x = self.maxp(x)
#         x = self.conv4(x) # (32, 128, 13, 14)
#         x = self.maxp(x)
#         x = self.convextra(x)
#
#         x = self.conv5(x)  # (32, 256, 6, 7)
#         x = self.bn5(x)
#         x = F.relu(x)
#
#         # flatten
#         shape = x.shape
#         x = x.view(*x.shape[:-2],-1) # *的作用是解包,前面两个维度不变，后面合并
#
#         x = self.gmlp(x)
#         x = F.relu(x)
#
#         x = x.reshape(x.shape[0], -1)
#         x = self.fc(x)
#         return x

# # MBnet
# class GLAM(nn.Module):
#     def __init__(self, shape = (26,63), **kwargs):
#         super(GLAM, self).__init__()
#         self.conva1 = MobileNetv2(in_channels=1, out_channels=16, kernel_size=(1, 3), stride=1, padding=(0, 1))
#         self.batchnormb1 = nn.BatchNorm2d(16)
#         self.conva2 = MobileNetv2(in_channels=1, out_channels=16, kernel_size=(3, 1), stride=1, padding=(1, 0))
#         self.batchnormb2 = nn.BatchNorm2d(16)
#         self.MSB1 = ResMultiConv(16)
#         self.maxp1 = nn.MaxPool2d(kernel_size=(2, 2))
#         self.MSB2 = ResMultiConv(32)
#         self.maxp2 = nn.MaxPool2d(kernel_size=(2, 2))
#         self.MSB3 = ResMultiConv(64)
#         self.conv3 = MobileNetv2(in_channels=128, out_channels=128, kernel_size=(5, 5), stride=1, padding=2, shortcut=1)
#         self.bn = nn.BatchNorm2d(128)
#         dim = (shape[0]//2) * (shape[1]//4)
#         self.gmlp = gMLP(dim=dim, depth=1, seq_len=128, act=nn.Tanh())
#         self.fc = nn.Linear(in_features=dim*128, out_features=4)
#
#     def forward(self, *input):
#         # 输入数据的大小：[32, 1, 26, 57]
#         xa = self.conva1(input[0])
#         xa = self.batchnormb1(xa)
#
#         xb = self.conva2(input[0])
#         xb = self.batchnormb2(xb)
#
#         xa = F.relu(xa) # [32, 16, 26, 57]
#         xb = F.relu(xb) # [32, 16, 26, 57]
#
#         x = torch.cat([xa, xb], dim = 2) # [32, 16, 52, 57]
# #         x = self.MSB1(x)
# #         x = self.maxp1(x)
# #         x = self.MSB2(x)
# #         x = self.maxp2(x)
# #         x = self.MSB3(x) # [32, 128, 13, 14]
# #         x = self.conv3(x)
# #         x = self.bn(x)
# #
# #         x = F.relu(x)
# #         x = x.view(*x.shape[:-2], -1)
# #
# #         x = self.gmlp(x)
# #         x = x.reshape(x.shape[0], -1)
# #         x = self.fc(x)
# #         return x

# # RepVGG
# class GLAM(nn.Module):
#     def __init__(self, shape = (26,63), **kwargs):
#         super(GLAM, self).__init__()
#         self.conva1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(1, 3), stride=1, padding=(0, 1))
#         self.batchnormb1 = nn.BatchNorm2d(16)
#         self.conva2 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(3, 1), stride=1, padding=(1, 0))
#         self.batchnormb2 = nn.BatchNorm2d(16)
#         self.MSB1 = rv.RepVGGplusBlock(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
#         self.maxp1 = nn.MaxPool2d(kernel_size=(2, 2))
#         self.MSB2 = rv.RepVGGplusBlock(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
#         self.maxp2 = nn.MaxPool2d(kernel_size=(2, 2))
#         self.MSB3 = rv.RepVGGplusBlock(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
#         self.conv3 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(5, 5), stride=1, padding=2)
#         self.bn = nn.BatchNorm2d(128)
#         dim = (shape[0]//2) * (shape[1]//4)
#         self.gmlp = gMLP(dim=dim, depth=1, seq_len=128, act=nn.Tanh())
#         self.fc = nn.Linear(in_features=dim*128, out_features=4)
#
#     def forward(self, *input):
#         # 输入数据的大小：[32, 1, 26, 57]
#         xa = self.conva1(input[0])
#         xa = self.batchnormb1(xa)
#
#         xb = self.conva2(input[0])
#         xb = self.batchnormb2(xb)
#
#         xa = F.relu(xa) # [32, 16, 26, 57]
#         xb = F.relu(xb) # [32, 16, 26, 57]
#
#         x = torch.cat([xa, xb], dim = 2) # [32, 16, 52, 57]
#         x = self.MSB1(x)
#         x = self.maxp1(x)
#         x = self.MSB2(x)
#         x = self.maxp2(x)
#         x = self.MSB3(x) # [32, 128, 13, 14]
#         x = self.conv3(x)
#         x = self.bn(x)
#
#         x = F.relu(x)
#         x = x.view(*x.shape[:-2], -1)
#
#         # x = self.gmlp(x)
#         x = x.reshape(x.shape[0], -1)
#         x = self.fc(x)
#         return x


# # RepVGG & FPN
# class GLAM(nn.Module):
#     def __init__(self, shape = (26,63), **kwargs):
#         super(GLAM, self).__init__()
#         self.features = []
#         self.inchannels = []
#         self.outchannels = 256
#         self.conva1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(1, 3), stride=1, padding=(0, 1))
#         self.batchnormb1 = nn.BatchNorm2d(16)
#         self.conva2 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(3, 1), stride=1, padding=(1, 0))
#         self.batchnormb2 = nn.BatchNorm2d(16)
#         self.MSB1 = rv.RepVGGplusBlock(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
#         self.maxp1 = nn.MaxPool2d(kernel_size=(2, 2))
#         self.MSB2 = rv.RepVGGplusBlock(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
#         self.maxp2 = nn.MaxPool2d(kernel_size=(2, 2))
#         self.MSB3 = rv.RepVGGplusBlock(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
#         self.conv3 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(5, 5), stride=1, padding=2)
#         self.bn = nn.BatchNorm2d(128)
#         dim = (shape[0]//2) * (shape[1]//4)
#         self.gmlp = gMLP(dim=dim, depth=1, seq_len=128, act=nn.Tanh())
#         self.fc = nn.Linear(in_features=dim*128, out_features=4)
#
#     def forward(self, *input):
#         # 输入数据的大小：[32, 1, 26, 57]
#         xa = self.conva1(input[0])
#         xa = self.batchnormb1(xa)
#
#         xb = self.conva2(input[0])
#         xb = self.batchnormb2(xb)
#
#         xa = F.relu(xa) # [32, 16, 26, 57]
#         xb = F.relu(xb) # [32, 16, 26, 57]
#
#         x = torch.cat([xa, xb], dim = 2)
#         self.features.append(x) # [32, 16, 52, 57]
#         self.inchannels.append(x.size()[1])
#
#         x = self.MSB1(x)
#         x = self.maxp1(x)
#         self.features.append(x) # [32, 32, 26, 28]
#         self.inchannels.append(x.size()[1])
#
#         x = self.MSB2(x)
#         x = self.maxp2(x)
#         self.features.append(x)  # [32, 64, 13, 14]
#         self.inchannels.append(x.size()[1])
#
#         x = self.MSB3(x)
#         x = self.conv3(x)
#         x = self.bn(x)
#
#         x = F.relu(x)
#         self.outchannels = x.size()[1]
#         self.features.append(x) # [32, 128, 6, 7]
#         self.inchannels.append(x.size()[1])
#         self.fpn = FPN.FPN(self.inchannels, self.outchannels)
#         x = self.fpn(self.features)
#
#         x = self.gmlp(x)
#         x = x.reshape(x.shape[0], -1)
#         x = self.fc(x)
#         return x


# RepVGG & PAN
class GLAM(nn.Module):
    def __init__(self, shape = (26,63), **kwargs):
        super(GLAM, self).__init__()
        self.features = []
        self.inchannels = []
        self.outchannels = 256
        self.conva1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(1, 3), stride=1, padding=(0, 1))
        self.batchnormb1 = nn.BatchNorm2d(16)
        self.conva2 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(3, 1), stride=1, padding=(1, 0))
        self.batchnormb2 = nn.BatchNorm2d(16)
        self.MSB1 = rv.RepVGGplusBlock(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.maxp1 = nn.MaxPool2d(kernel_size=(2, 2))
        self.MSB2 = rv.RepVGGplusBlock(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.maxp2 = nn.MaxPool2d(kernel_size=(2, 2))
        self.MSB3 = rv.RepVGGplusBlock(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(5, 5), stride=1, padding=2)
        self.maxp3 = nn.MaxPool2d(kernel_size=(2, 2))
        self.bn = nn.BatchNorm2d(128)
        dim = (shape[0]//2) * (shape[1]//4)
        self.gmlp = gMLP(dim=3916, depth=1, seq_len=128, act=nn.Tanh())
        self.fc = nn.Linear(in_features=3916*128, out_features=4)

    def forward(self, *input):
        # 输入数据的大小：[32, 1, 26, 57]
        xa = self.conva1(input[0])
        xa = self.batchnormb1(xa)

        xb = self.conva2(input[0])
        xb = self.batchnormb2(xb)

        xa = F.relu(xa) # [32, 16, 26, 57]
        xb = F.relu(xb) # [32, 16, 26, 57]

        x = torch.cat([xa, xb], dim = 2)
        self.features.append(x) # [32, 16, 52, 57]
        self.inchannels.append(x.size()[1])

        x = self.MSB1(x)
        x = self.maxp1(x)
        self.features.append(x) # [32, 32, 26, 28]
        self.inchannels.append(x.size()[1])

        x = self.MSB2(x)
        x = self.maxp2(x)
        self.features.append(x)  # [32, 64, 13, 14]
        self.inchannels.append(x.size()[1])

        x = self.MSB3(x)
        x = self.conv3(x)
        x = self.maxp3(x)
        x = self.bn(x)

        x = F.relu(x)
        self.outchannels = x.size()[1]
        self.features.append(x) # [32, 128, 6, 7]
        self.inchannels.append(x.size()[1])
        self.pan = PANet.PAN(self.inchannels, self.outchannels)
        x = self.pan(self.features)

        # x = self.gmlp(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        return x



# # original
# class GLAM(nn.Module):
#     def __init__(self, shape = (26,63), **kwargs):
#         super(GLAM, self).__init__()
#         self.conva1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(1, 3), stride=1, padding=(0, 1))
#         self.batchnormb1 = nn.BatchNorm2d(16)
#         self.conva2 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(3, 1), stride=1, padding=(1, 0))
#         self.batchnormb2 = nn.BatchNorm2d(16)
#         self.MSB1 = ResMultiConv(16)
#         self.maxp1 = nn.MaxPool2d(kernel_size=(2, 2))
#         self.MSB2 = ResMultiConv(32)
#         self.maxp2 = nn.MaxPool2d(kernel_size=(2, 2))
#         self.MSB3 = ResMultiConv(64)
#         self.conv3 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(5, 5), stride=1, padding=2)
#         self.bn = nn.BatchNorm2d(128)
#         dim = (shape[0]//2) * (shape[1]//4)
#         self.gmlp = gMLP(dim=dim, depth=1, seq_len=128, act=nn.Tanh())
#         self.fc = nn.Linear(in_features=dim*128, out_features=4)
#
#     def forward(self, *input):
#         # 输入数据的大小：[32, 1, 26, 57]
#         xa = self.conva1(input[0])
#         xa = self.batchnormb1(xa)
#
#         xb = self.conva2(input[0])
#         xb = self.batchnormb2(xb)
#
#         xa = F.relu(xa) # [32, 16, 26, 57]
#         xb = F.relu(xb) # [32, 16, 26, 57]
#
#         x = torch.cat([xa, xb], dim = 2) # [32, 16, 52, 57]
#         x = self.MSB1(x)
#         x = self.maxp1(x)
#         x = self.MSB2(x)
#         x = self.maxp2(x)
#         x = self.MSB3(x) # [32, 128, 13, 14]
#         x = self.conv3(x)
#         x = self.bn(x)
#
#         x = F.relu(x)
#         x = x.view(*x.shape[:-2], -1)
#
#         x = self.gmlp(x)
#         x = x.reshape(x.shape[0], -1)
#         x = self.fc(x)
#         return x

# res
# class GLAM(nn.Module):
#     def __init__(self, shape = (26,63), **kwargs):
#         super(GLAM, self).__init__()
#         self.conva1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(1, 3), stride=1, padding=(0, 1))
#         self.batchnormb1 = nn.BatchNorm2d(16)
#         self.conva2 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(3, 1), stride=1, padding=(1, 0))
#         self.batchnormb2 = nn.BatchNorm2d(16)
#         self.MSB1 = ResMultiConv(16)
#         self.maxp1 = nn.MaxPool2d(kernel_size=(2, 2))
#         self.MSB2 = ResMultiConv(32)
#         self.maxp2 = nn.MaxPool2d(kernel_size=(2, 2))
#         self.MSB3 = ResMultiConv(64)
#         self.conv3 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(5, 5), stride=1, padding=2)
#         self.bn = nn.BatchNorm2d(128)
#         self.convy = nn.Conv2d(in_channels=16, out_channels=128,kernel_size=(5, 5), stride=4, padding=(2, 0))
#         dim = (shape[0]//2) * (shape[1]//4)
#         self.gmlp = gMLP(dim=dim, depth=1, seq_len=128, act=nn.Tanh())
#         self.fc = nn.Linear(in_features=dim*128, out_features=4)
#
#     def forward(self, *input):
#         # 输入数据的大小：[32, 1, 26, 57]
#         xa = self.conva1(input[0])
#         xa = self.batchnormb1(xa)
#
#         xb = self.conva2(input[0])
#         xb = self.batchnormb2(xb)
#
#         xa = F.relu(xa) # [32, 16, 26, 57]
#         xb = F.relu(xb) # [32, 16, 26, 57]
#
#         x = torch.cat([xa, xb], dim = 2) # [32, 16, 52, 57]
#         y = self.convy(x)
#         x = self.MSB1(x)
#         x = self.maxp1(x)
#         x = self.MSB2(x)
#         x = self.maxp2(x)
#         x = self.MSB3(x) # [32, 128, 13, 14]
#         x = self.conv3(x)
#         x = x + y
#         x = self.bn(x)
#
#         x = F.relu(x)
#         x = x.view(*x.shape[:-2], -1)
#
#         x = self.gmlp(x)
#         x = x.reshape(x.shape[0], -1)
#         x = self.fc(x)
#         return x

class gMLPResConv3(nn.Module):
    '''
    GLAM - Multiscale
    '''
    def __init__(self, shape=(26,63), **kwargs):
        super(gMLPResConv3, self).__init__()
        self.conv1a = nn.Conv2d(kernel_size=(3, 1), in_channels=1, out_channels=16, padding=(1, 0))
        self.conv1b = nn.Conv2d(kernel_size=(1, 3), in_channels=1, out_channels=16, padding=(0, 1))
        self.conv2 = ResConv3(16)
        self.conv3 = ResConv3(32)
        self.conv4 = ResConv3(64)
        self.conv5 = nn.Conv2d(kernel_size=(5, 5), in_channels=128, out_channels=128, padding=2)
        self.maxp = nn.MaxPool2d(kernel_size=(2, 2))
        self.bn1a = nn.BatchNorm2d(16)
        self.bn1b = nn.BatchNorm2d(16)
        self.bn5 = nn.BatchNorm2d(128)
        dim = (shape[0]//2) * (shape[1]//4)
        i = 128 * dim
        self.fc = nn.Linear(in_features=i, out_features=4)
        self.dropout = nn.Dropout(0.5)

        self.gmlp = gMLP(dim = dim, depth = 1, seq_len = 128, act = nn.Tanh())

    def forward(self, *input):
        # input[0]: torch.Size([32, 1, 26, 63])
        xa = self.conv1a(input[0]) # (32, 16, 25, 62)
        xa = self.bn1a(xa) # (32, 16, 25, 62)

        xa = F.relu(xa)
        xb = self.conv1b(input[0])
        xb = self.bn1b(xb)

        xb = F.relu(xb)
        x = torch.cat((xa, xb), 2) # (32, 16, 50, 62)

        x = self.conv2(x) # (32, 32, 50, 62)
        x = self.maxp(x)
        x = self.conv3(x) # (32, 64, 25, 31)
        x = self.maxp(x)
        x = self.conv4(x) # (32, 128, 12, 15)

        x = self.conv5(x)  # (32, 128, 12, 15)
        x = self.bn5(x)
        x = F.relu(x)

        # flatten
        shape = x.shape
        x = x.view(*x.shape[:-2],-1)

        x = self.gmlp(x)
        x = F.relu(x)

        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        return x

class gMLPResConv5(nn.Module):
    '''
    GLAM5 - Multiscale
    '''
    def __init__(self, shape=(26,63), **kwargs):
        super(gMLPResConv5, self).__init__()
        self.conv1a = nn.Conv2d(kernel_size=(3, 1), in_channels=1, out_channels=16, padding=(1, 0))
        self.conv1b = nn.Conv2d(kernel_size=(1, 3), in_channels=1, out_channels=16, padding=(0, 1))
        self.conv2 = ResConv5(16)
        self.conv3 = ResConv5(32)
        self.conv4 = ResConv5(64)
        self.conv5 = nn.Conv2d(kernel_size=(5, 5), in_channels=128, out_channels=128, padding=2)
        self.maxp = nn.MaxPool2d(kernel_size=(2, 2))
        self.bn1a = nn.BatchNorm2d(16)
        self.bn1b = nn.BatchNorm2d(16)
        self.bn5 = nn.BatchNorm2d(128)
        dim = (shape[0]//2) * (shape[1]//4)
        i = 128 * dim
        self.fc = nn.Linear(in_features=i, out_features=4)
        self.dropout = nn.Dropout(0.5)

        self.gmlp = gMLP(dim = dim, depth = 1, seq_len = 128, act = nn.Tanh())

    def forward(self, *input):
        # input[0]: torch.Size([32, 1, 26, 63])
        xa = self.conv1a(input[0]) # (32, 16, 25, 62)
        xa = self.bn1a(xa) # (32, 16, 25, 62)

        xa = F.relu(xa)
        xb = self.conv1b(input[0])
        xb = self.bn1b(xb)

        xb = F.relu(xb)
        x = torch.cat((xa, xb), 2) # (32, 16, 50, 62)

        x = self.conv2(x) # (32, 32, 50, 62)
        x = self.maxp(x)
        x = self.conv3(x) # (32, 64, 25, 31)
        x = self.maxp(x)
        x = self.conv4(x) # (32, 128, 12, 15)

        x = self.conv5(x)  # (32, 128, 12, 15)
        x = self.bn5(x)
        x = F.relu(x)

        # flatten
        shape = x.shape
        x = x.view(*x.shape[:-2],-1)

        x = self.gmlp(x)
        x = F.relu(x)

        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        return x

class gMLPMultiConv(nn.Module):
    '''
    GLAM - Resnet
    '''
    def __init__(self, shape=(26,63), **kwargs):
        super(gMLPMultiConv, self).__init__()
        self.conv1a = nn.Conv2d(kernel_size=(3, 1), in_channels=1, out_channels=16, padding=(1, 0))
        self.conv1b = nn.Conv2d(kernel_size=(1, 3), in_channels=1, out_channels=16, padding=(0, 1))
        self.conv2 = MultiConv(16)
        self.conv3 = MultiConv(32)
        self.conv4 = MultiConv(64)
        self.conv5 = nn.Conv2d(kernel_size=(5, 5), in_channels=128, out_channels=128, padding=2)
        self.maxp = nn.MaxPool2d(kernel_size=(2, 2))
        self.bn1a = nn.BatchNorm2d(16)
        self.bn1b = nn.BatchNorm2d(16)
        self.bn5 = nn.BatchNorm2d(128)
        dim = (shape[0]//2) * (shape[1]//4)
        i = 128 * dim
        self.fc = nn.Linear(in_features=i, out_features=4)
        self.dropout = nn.Dropout(0.5)

        self.gmlp = gMLP(dim = dim, depth = 1, seq_len = 128, act = nn.Tanh())

    def forward(self, *input):
        # input[0]: torch.Size([32, 1, 26, 63])
        xa = self.conv1a(input[0]) # (32, 16, 25, 62)
        xa = self.bn1a(xa) # (32, 16, 25, 62)

        xa = F.relu(xa)
        xb = self.conv1b(input[0])
        xb = self.bn1b(xb)

        xb = F.relu(xb)
        x = torch.cat((xa, xb), 2) # (32, 16, 50, 62)

        x = self.conv2(x) # (32, 32, 50, 62)
        x = self.maxp(x)
        x = self.conv3(x) # (32, 64, 25, 31)
        x = self.maxp(x)
        x = self.conv4(x) # (32, 128, 12, 15)

        x = self.conv5(x)  # (32, 128, 12, 15)
        x = self.bn5(x)
        x = F.relu(x)

        # flatten
        shape = x.shape
        x = x.view(*x.shape[:-2],-1)

        x = self.gmlp(x)
        x = F.relu(x)

        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        return x

class MHResMultiConv3(nn.Module):
    '''
    Multi-Head-Attention with Multiscale blocks
    '''
    def __init__(self, head=4, attn_hidden=64, shape=(26,63), **kwargs):
        super(MHResMultiConv3, self).__init__()
        self.head = head
        self.attn_hidden = attn_hidden
        self.conv1a = nn.Conv2d(kernel_size=(3, 1), in_channels=1, out_channels=16, padding=(1, 0))
        self.conv1b = nn.Conv2d(kernel_size=(1, 3), in_channels=1, out_channels=16, padding=(0, 1))
        self.conv2 = ResMultiConv(16)
        self.conv3 = ResMultiConv(32)
        self.conv4 = ResMultiConv(64)
        self.conv5 = nn.Conv2d(kernel_size=(5, 5), in_channels=128, out_channels=128, padding=2)
        self.maxp = nn.MaxPool2d(kernel_size=(2, 2))
        self.bn1a = nn.BatchNorm2d(16)
        self.bn1b = nn.BatchNorm2d(16)
        self.bn5 = nn.BatchNorm2d(128)
        i = self.attn_hidden * self.head * (shape[0]//2) * (shape[1]//4)
        self.fc = nn.Linear(in_features=i, out_features=4)

        self.dropout = nn.Dropout(0.5)
        self.attention_query = nn.ModuleList()
        self.attention_key = nn.ModuleList()
        self.attention_value = nn.ModuleList()

        for i in range(self.head):
            self.attention_query.append(nn.Conv2d(in_channels=128, out_channels=self.attn_hidden, kernel_size=1))
            self.attention_key.append(nn.Conv2d(in_channels=128, out_channels=self.attn_hidden, kernel_size=1))
            self.attention_value.append(nn.Conv2d(in_channels=128, out_channels=self.attn_hidden, kernel_size=1))

    def forward(self, *input):
        # input[0]: torch.Size([32, 1, 26, 63])
        xa = self.conv1a(input[0]) # (32, 16, 25, 62)
        xa = self.bn1a(xa) # (32, 16, 25, 62)

        xa = F.relu(xa)
        xb = self.conv1b(input[0])
        xb = self.bn1b(xb)

        xb = F.relu(xb)
        x = torch.cat((xa, xb), 2) # (32, 16, 50, 62)

        x = self.conv2(x) # (32, 32, 50, 62)
        x = self.maxp(x)
        x = self.conv3(x) # (32, 64, 25, 31)
        x = self.maxp(x)
        x = self.conv4(x) # (32, 128, 12, 15)

        x = self.conv5(x)  # (32, 128, 12, 15)
        x = self.bn5(x)
        x = F.relu(x)

        attn = None
        for i in range(self.head):
            Q = self.attention_query[i](x)
            K = self.attention_key[i](x)
            V = self.attention_value[i](x)
            attention = F.softmax(torch.mul(Q, K),dim=1)
            attention = torch.mul(attention, V)

            if (attn is None):
                attn = attention
            else:
                attn = torch.cat((attn, attention), 2)
        x = attn
        x = F.relu(x)

        x = x.reshape(x.shape[0], x.shape[1] * x.shape[2] * x.shape[3])

        x = self.fc(x)
        return x

class AAResMultiConv3(nn.Module):
    '''
    Area Attention with Multiscale blocks
    '''
    def __init__(self, head=4, attn_hidden=64, shape=(26,63), **kwargs):
        super(AAResMultiConv3, self).__init__()
        self.head = head
        self.attn_hidden = attn_hidden
        self.conv1a = nn.Conv2d(kernel_size=(3, 1), in_channels=1, out_channels=16, padding=(1, 0))
        self.conv1b = nn.Conv2d(kernel_size=(1, 3), in_channels=1, out_channels=16, padding=(0, 1))
        self.conv2 = ResMultiConv(16)
        self.conv3 = ResMultiConv(32)
        self.conv4 = ResMultiConv(64)
        self.conv5 = nn.Conv2d(kernel_size=(5, 5), in_channels=128, out_channels=128, padding=2)
        self.maxp = nn.MaxPool2d(kernel_size=(2, 2))
        self.bn1a = nn.BatchNorm2d(16)
        self.bn1b = nn.BatchNorm2d(16)
        self.bn5 = nn.BatchNorm2d(128)

        i = 128 * (shape[0]//2) * (shape[1]//4) 
        self.fc = nn.Linear(in_features=i, out_features=4)
        # self.dropout = nn.Dropout(0.5)

        self.area_attention = AreaAttention(
            key_query_size=80,
            area_key_mode='mean',
            area_value_mode='sum',
            max_area_height=3,
            max_area_width=3,
            dropout_rate=0.5,
        )

    def forward(self, *input):
        # input[0]: torch.Size([32, 1, 26, 63])
        xa = self.conv1a(input[0]) # (32, 16, 25, 62)
        xa = self.bn1a(xa) # (32, 16, 25, 62)

        xa = F.relu(xa)
        xb = self.conv1b(input[0])
        xb = self.bn1b(xb)

        xb = F.relu(xb)
        x = torch.cat((xa, xb), 2) # (32, 16, 50, 62)

        x = self.conv2(x) # (32, 32, 50, 62)
        x = self.maxp(x)
        x = self.conv3(x) # (32, 64, 25, 31)
        x = self.maxp(x)
        x = self.conv4(x) # (32, 128, 12, 15)

        x = self.conv5(x)  # (32, 128, 12, 15)
        x = self.bn5(x)
        x = F.relu(x)

        shape = x.shape
        x = x.contiguous().permute(0, 2, 3, 1).view(shape[0], shape[3]*shape[2], shape[1])
        x = self.area_attention(x,x,x)
        x = F.relu(x)
        x = x.reshape(x.shape[0], -1)

        x = self.fc(x)
        
        return x

class ResMultiConv3(nn.Module):
    '''
    GLAM - gMLP
    '''
    def __init__(self, shape=(26,63), **kwargs):
        super(ResMultiConv3, self).__init__()
        self.conv1a = nn.Conv2d(kernel_size=(3, 1), in_channels=1, out_channels=16, padding=(1, 0))
        self.conv1b = nn.Conv2d(kernel_size=(1, 3), in_channels=1, out_channels=16, padding=(0, 1))
        self.conv2 = ResMultiConv(16)
        self.conv3 = ResMultiConv(32)
        self.conv4 = ResMultiConv(64)
        self.conv5 = nn.Conv2d(kernel_size=(5, 5), in_channels=128, out_channels=128, padding=2)
        self.maxp = nn.MaxPool2d(kernel_size=(2, 2))
        self.bn1a = nn.BatchNorm2d(16)
        self.bn1b = nn.BatchNorm2d(16)
        self.bn5 = nn.BatchNorm2d(128)
        dim = (shape[0]//2) * (shape[1]//4)
        i = 128 * dim
        self.fc = nn.Linear(in_features=i, out_features=4)
        self.dropout = nn.Dropout(0.5)

    def forward(self, *input):
        # input[0]: torch.Size([32, 1, 26, 63])
        xa = self.conv1a(input[0]) # (32, 16, 25, 62)
        xa = self.bn1a(xa) # (32, 16, 25, 62)

        xa = F.relu(xa)
        xb = self.conv1b(input[0])
        xb = self.bn1b(xb)

        xb = F.relu(xb)
        x = torch.cat((xa, xb), 2) # (32, 16, 50, 62)

        x = self.conv2(x) # (32, 32, 50, 62)
        x = self.maxp(x)
        x = self.conv3(x) # (32, 64, 25, 31)
        x = self.maxp(x)
        x = self.conv4(x) # (32, 128, 12, 15)

        x = self.conv5(x)  # (32, 128, 12, 15)
        x = self.bn5(x)
        x = F.relu(x)

        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        return x

class ResMultiConv5(nn.Module):
    '''
    GLAM5 - gMLP
    '''
    def __init__(self, shape=(26,63), **kwargs):
        super(ResMultiConv5, self).__init__()
        self.conv1a = nn.Conv2d(kernel_size=(5, 1), in_channels=1, out_channels=16, padding=(2, 0))
        self.conv1b = nn.Conv2d(kernel_size=(1, 5), in_channels=1, out_channels=16, padding=(0, 2))
        self.conv2 = ResMultiConv(16)
        self.conv3 = ResMultiConv(32)
        self.conv4 = ResMultiConv(64)
        self.conv5 = nn.Conv2d(kernel_size=(5, 5), in_channels=128, out_channels=128, padding=2)
        self.maxp = nn.MaxPool2d(kernel_size=(2, 2))
        self.bn1a = nn.BatchNorm2d(16)
        self.bn1b = nn.BatchNorm2d(16)
        self.bn5 = nn.BatchNorm2d(128)
        dim = (shape[0]//2) * (shape[1]//4)
        i = 128 * dim
        self.fc = nn.Linear(in_features=i, out_features=4)
        self.dropout = nn.Dropout(0.5)

    def forward(self, *input):
        # input[0]: torch.Size([32, 1, 26, 63])
        xa = self.conv1a(input[0]) # (32, 16, 25, 62)
        xa = self.bn1a(xa) # (32, 16, 25, 62)

        xa = F.relu(xa)
        xb = self.conv1b(input[0])
        xb = self.bn1b(xb)

        xb = F.relu(xb)
        x = torch.cat((xa, xb), 2) # (32, 16, 50, 62)

        x = self.conv2(x) # (32, 32, 50, 62)
        x = self.maxp(x)
        x = self.conv3(x) # (32, 64, 25, 31)
        x = self.maxp(x)
        x = self.conv4(x) # (32, 128, 12, 15)

        x = self.conv5(x)  # (32, 128, 12, 15)
        x = self.bn5(x)
        x = F.relu(x)

        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        return x

class gMLPgResMultiConv3(nn.Module):
    def __init__(self, shape=(26,63), **kwargs):
        super(gMLPgResMultiConv3, self).__init__()
        self.conv1a = nn.Conv2d(kernel_size=(3, 1), in_channels=1, out_channels=16, padding=(1, 0))
        self.conv1b = nn.Conv2d(kernel_size=(1, 3), in_channels=1, out_channels=16, padding=(0, 1))
        self.conv2 = ResMultiConv(16)
        self.conv3 = ResMultiConv(32)
        self.conv4 = ResMultiConv(64)
        self.conv5 = nn.Conv2d(kernel_size=(5, 5), in_channels=128, out_channels=128, padding=2)
        self.maxp = nn.MaxPool2d(kernel_size=(2, 2))
        self.bn1a = nn.BatchNorm2d(16)
        self.bn1b = nn.BatchNorm2d(16)
        self.bn5 = nn.BatchNorm2d(128)
        dim = (shape[0]//4) * (shape[1]//4)
        i = 128 * dim
        self.fc = nn.Linear(in_features=i, out_features=4)
        self.dropout = nn.Dropout(0.5)

        self.gmlp = gMLP(dim = dim, depth = 1, seq_len = 128, act = nn.Tanh())
        self.sgu = SpatialGatingUnit(dim = shape[0] * shape[1] * 2, dim_seq = 16, act = nn.Tanh())

    def forward(self, *input):
        # input[0]: torch.Size([32, 1, 26, 63])
        xa = self.conv1a(input[0]) # (32, 16, 26, 63)
        xa = self.bn1a(xa) # (32, 16, 26, 63)

        xa = F.relu(xa)
        xb = self.conv1b(input[0])
        xb = self.bn1b(xb)

        xb = F.relu(xb)
        x = torch.cat((xa, xb), 2) # (32, 16, 50, 62)

        shape = x.shape
        x = x.view(*x.shape[:-2],-1)
        x = self.sgu(x)
        x = x.view(shape[0], shape[1], shape[2]//2, shape[3])

        x = self.conv2(x) # (32, 32, 50, 62)
        x = self.maxp(x)
        x = self.conv3(x) # (32, 64, 25, 31)
        x = self.maxp(x)
        x = self.conv4(x) # (32, 128, 12, 15)

        x = self.conv5(x)  # (32, 128, 12, 15)
        x = self.bn5(x)
        x = F.relu(x)

        # flatten
        shape = x.shape
        x = x.view(*x.shape[:-2],-1)

        x = self.gmlp(x)
        x = F.relu(x)

        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        return x

class aMLPResMultiConv3(nn.Module):
    def __init__(self, shape=(26,63), **kwargs):
        super(aMLPResMultiConv3, self).__init__()
        self.conv1a = nn.Conv2d(kernel_size=(3, 1), in_channels=1, out_channels=16, padding=(1, 0))
        self.conv1b = nn.Conv2d(kernel_size=(1, 3), in_channels=1, out_channels=16, padding=(0, 1))
        self.conv2 = ResMultiConv(16)
        self.conv3 = ResMultiConv(32)
        self.conv4 = ResMultiConv(64)
        self.conv5 = nn.Conv2d(kernel_size=(5, 5), in_channels=128, out_channels=128, padding=2)
        self.maxp = nn.MaxPool2d(kernel_size=(2, 2))
        self.bn1a = nn.BatchNorm2d(16)
        self.bn1b = nn.BatchNorm2d(16)
        self.bn5 = nn.BatchNorm2d(128)
        dim = (shape[0]//2) * (shape[1]//4)
        i = 128 * dim
        self.fc = nn.Linear(in_features=i, out_features=4)
        self.dropout = nn.Dropout(0.5)

        self.gmlp = gMLP(dim = dim, depth = 1, seq_len = 128, attn_dim = 64, act = nn.Tanh())

    def forward(self, *input):
        # input[0]: torch.Size([32, 1, 26, 63])
        xa = self.conv1a(input[0]) # (32, 16, 25, 62)
        xa = self.bn1a(xa) # (32, 16, 25, 62)

        xa = F.relu(xa)
        xb = self.conv1b(input[0])
        xb = self.bn1b(xb)

        xb = F.relu(xb)
        x = torch.cat((xa, xb), 2) # (32, 16, 50, 62)

        x = self.conv2(x) # (32, 32, 50, 62)
        x = self.maxp(x)
        x = self.conv3(x) # (32, 64, 25, 31)
        x = self.maxp(x)
        x = self.conv4(x) # (32, 128, 12, 15)

        x = self.conv5(x)  # (32, 128, 12, 15)
        x = self.bn5(x)
        x = F.relu(x)

        # flatten
        shape = x.shape
        x = x.view(*x.shape[:-2],-1)

        x = self.gmlp(x)
        x = F.relu(x)

        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        return x

class gMLPResMultiConv35(nn.Module):
    '''
    Temporal and Spatial convolution with multiscales
    '''
    def __init__(self, shape=(26,63), **kwargs):
        super(gMLPResMultiConv35, self).__init__()
        self.conv1a1 = nn.Conv2d(kernel_size=(3, 1), in_channels=1, out_channels=8, padding=(1, 0))
        self.conv1a2 = nn.Conv2d(kernel_size=(5, 1), in_channels=1, out_channels=8, padding=(2, 0))
        self.conv1b1 = nn.Conv2d(kernel_size=(1, 3), in_channels=1, out_channels=8, padding=(0, 1))
        self.conv1b2 = nn.Conv2d(kernel_size=(1, 5), in_channels=1, out_channels=8, padding=(0, 2))
        self.conv2 = ResMultiConv(16)
        self.conv3 = ResMultiConv(32)
        self.conv4 = ResMultiConv(64)
        self.conv5 = nn.Conv2d(kernel_size=(5, 5), in_channels=128, out_channels=128, padding=2)
        self.maxp = nn.MaxPool2d(kernel_size=(2, 2))
        self.bn1a = nn.BatchNorm2d(16)
        self.bn1b = nn.BatchNorm2d(16)
        self.bn5 = nn.BatchNorm2d(128)
        dim = (shape[0]//2) * (shape[1]//4)
        i = 128 * dim
        self.fc = nn.Linear(in_features=i, out_features=4)
        self.dropout = nn.Dropout(0.5)

        self.gmlp = gMLP(dim = dim, depth = 1, seq_len = 128, act = nn.Tanh())

    def forward(self, *input):
        # input[0]: torch.Size([32, 1, 26, 63])
        xa1 = self.conv1a1(input[0]) # (32, 8, 26, 63)
        xa2 = self.conv1a2(input[0]) # (32, 8, 26, 63)
        xa = torch.cat((xa1,xa2),1)
        xa = self.bn1a(xa) # (32, 16, 26, 63)
        xa = F.relu(xa)

        xb1 = self.conv1b1(input[0])
        xb2 = self.conv1b2(input[0])
        xb = torch.cat((xb1,xb2),1)
        xb = self.bn1b(xb)

        xb = F.relu(xb)
        x = torch.cat((xa, xb), 2) # (32, 16, 50, 62)

        x = self.conv2(x) # (32, 32, 50, 62)
        x = self.maxp(x)
        x = self.conv3(x) # (32, 64, 25, 31)
        x = self.maxp(x)
        x = self.conv4(x) # (32, 128, 12, 15)

        x = self.conv5(x)  # (32, 128, 12, 15)
        x = self.bn5(x)
        x = F.relu(x)

        # flatten
        shape = x.shape
        x = x.view(*x.shape[:-2],-1)

        x = self.gmlp(x)
        x = F.relu(x)

        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        return x

class SEBlock(nn.Module):

    def __init__(self, input_channels, internal_neurons):
        super(SEBlock, self).__init__()
        self.down = nn.Conv2d(in_channels=input_channels, out_channels=internal_neurons, kernel_size=1, stride=1, bias=True)
        self.up = nn.Conv2d(in_channels=internal_neurons, out_channels=input_channels, kernel_size=1, stride=1, bias=True)
        self.input_channels = input_channels

    def forward(self, inputs):
        x = F.avg_pool2d(inputs, kernel_size=(inputs.size(2), inputs.size(3)))
        x = self.down(x)
        x = F.relu(x)
        x = self.up(x)
        x = torch.sigmoid(x)
        x = x.view(-1, self.input_channels, 1, 1)
        return inputs * x

def conv_bn(in_channels, out_channels, kernel_size, stride, padding, groups=1):
    result = nn.Sequential()
    result.add_module('conv', nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                                  kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=False))
    result.add_module('bn', nn.BatchNorm2d(num_features=out_channels))
    return result

class RepVGGplusBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3,
                 stride=1, padding=2, dilation=1, groups=1, padding_mode='zeros',
                 deploy=False,
                 use_post_se=True):
        super(RepVGGplusBlock, self).__init__()
        self.deploy = deploy
        self.groups = groups
        self.in_channels = in_channels

        assert kernel_size == 3
        assert padding == 1

        self.nonlinearity = nn.ReLU()

        if use_post_se:
            self.post_se = SEBlock(out_channels, internal_neurons=out_channels // 4)
        else:
            self.post_se = nn.Identity()

        if deploy:
            self.rbr_reparam = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                                      padding=padding, dilation=dilation, groups=groups, bias=True, padding_mode=padding_mode)
        else:
            if out_channels == in_channels and stride == 1:
                self.rbr_identity = nn.BatchNorm2d(num_features=out_channels)
            else:
                self.rbr_identity = None
            self.rbr_dense = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups)
            padding_11 = padding - kernel_size // 2
            self.rbr_1x1 = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, padding=padding_11, groups=groups)

    def forward(self, x):
        if self.deploy:
            return self.post_se(self.nonlinearity(self.rbr_reparam(x)))

        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(x)
        out = self.rbr_dense(x) + self.rbr_1x1(x) + id_out
        out = self.post_se(self.nonlinearity(out))
        return out


    #   This func derives the equivalent kernel and bias in a DIFFERENTIABLE way.
    #   You can get the equivalent kernel and bias at any time and do whatever you want,
    #   for example, apply some penalties or constraints during training, just like you do to the other models.
    #   May be useful for quantization or pruning.
    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        kernelid, biasid = self._fuse_bn_tensor(self.rbr_identity)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0
        if isinstance(branch, nn.Sequential):
            #   For the 1x1 or 3x3 branch
            kernel, running_mean, running_var, gamma, beta, eps = branch.conv.weight, branch.bn.running_mean, branch.bn.running_var, branch.bn.weight, branch.bn.bias, branch.bn.eps
        else:
            #   For the identity branch
            assert isinstance(branch, nn.BatchNorm2d)
            if not hasattr(self, 'id_tensor'):
                #   Construct and store the identity kernel in case it is used multiple times
                input_dim = self.in_channels // self.groups
                kernel_value = np.zeros((self.in_channels, input_dim, 3, 3), dtype=np.float32)
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel, running_mean, running_var, gamma, beta, eps = self.id_tensor, branch.running_mean, branch.running_var, branch.weight, branch.bias, branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def switch_to_deploy(self):
        if hasattr(self, 'rbr_reparam'):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.rbr_reparam = nn.Conv2d(in_channels=self.rbr_dense.conv.in_channels,
                                     out_channels=self.rbr_dense.conv.out_channels,
                                     kernel_size=self.rbr_dense.conv.kernel_size, stride=self.rbr_dense.conv.stride,
                                     padding=self.rbr_dense.conv.padding, dilation=self.rbr_dense.conv.dilation,
                                     groups=self.rbr_dense.conv.groups, bias=True)
        self.rbr_reparam.weight.data = kernel
        self.rbr_reparam.bias.data = bias
        self.__delattr__('rbr_dense')
        self.__delattr__('rbr_1x1')
        if hasattr(self, 'rbr_identity'):
            self.__delattr__('rbr_identity')
        if hasattr(self, 'id_tensor'):
            self.__delattr__('id_tensor')
        self.deploy = True

def repvgg_model_convert(model:torch.nn.Module, save_path=None, do_copy=True):
    import copy
    if do_copy:
        model = copy.deepcopy(model)
    for module in model.modules():
        if hasattr(module, 'switch_to_deploy'):
            module.switch_to_deploy()
    if save_path is not None:
        torch.save(model.state_dict(), save_path)
    return model


if __name__ == '__main__':
    input_data = torch.randn(32, 1, 26, 57)
    model = GLAM((26, 57))
    output_data = model(input_data)