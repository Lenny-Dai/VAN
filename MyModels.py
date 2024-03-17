import torch
import torch.nn as nn
import torch.nn.functional as F
from area_attention import AreaAttention
from g_mlp_pytorch import gMLP
from g_mlp_pytorch import SpatialGatingUnit

class ResMultiConv(nn.Module):
    def __init__(self, channels = 16, **kwargs):
        super(ResMultiConv, self).__init__()
        self.conv3 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv3d(in_channels=channels, out_channels=channels, kernel_size=5, stride=1, padding=2)
        self.batchnormb1 = nn.BatchNorm2d(channels)
        self.batchnormb2 = nn.BatchNorm2d(channels)

    def forward(self, *inputs):
        xa = self.conv3(inputs[0]) + inputs[0]
        xa = self.batchnormb1(xa)
        xa = F.relu(xa)

        xb = self.conv5(inputs[0]) + inputs[0]
        xb = self.batchnormb2(xb)
        xb = F.relu(xb)

        return torch.cat([xa, xb], dim = 1)

class GLAM(nn.Module):
    def __init__(self, shape = (26,63), **kwargs):
        super(GLAM, self).__init__()
        self.conva1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(1, 3), stride=1, padding=(0, 1))
        self.batchnormb1 = nn.BatchNorm2d(16)
        self.conva2 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(3, 1), stride=1, padding=(1, 0))
        self.batchnormb2 = nn.BatchNorm2d(16)
        self.MSB1 = ResMultiConv(16)
        self.maxp1 = nn.MaxPool2d(kernel_size=(2, 2))
        self.MSB2 = ResMultiConv(32)
        self.maxp2 = nn.MaxPool2d(kernel_size=(2, 2))
        self.MSB3 = ResMultiConv(64)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(5, 5), stride=1, padding=2)
        self.bn = nn.BatchNorm2d(128)
        dim = (shape[0]//2) * (shape[1]//4)
        self.gmlp = gMLP(dim=dim, depth=1, seq_len=128, act=nn.Tanh())
        self.fc = nn.Linear(in_features=dim*128, out_features=4)

    def forward(self, *input):
        # 输入数据的大小：[32, 1, 26, 57]
        xa = self.conva1(input[0])
        xa = self.batchnormb1(xa)

        xb = self.conva2(input[0])
        xb = self.batchnormb2(xb)

        xa = F.relu(xa) # [32, 16, 26, 57]
        xb = F.relu(xb) # [32, 16, 26, 57]

        x = torch.cat([xa, xb], dim = 2) # [32, 16, 52, 57]
        x = self.MSB1(x)
        x = self.maxp1(x)
        x = self.MSB2(x)
        x = self.maxp2(x)
        x = self.MSB3(x) # [32, 128, 13, 14]
        x = self.conv3(x)
        x = self.bn(x)

        x = F.relu(x)
        x = x.view(*x.shape[:-2], -1)

        x = x.self.gmlp(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        return x








