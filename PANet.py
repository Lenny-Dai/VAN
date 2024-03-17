import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import RepVGG as rv

class PAN(nn.Module):
    def __init__(self, in_channel_list, out_channel):
        super(PAN, self).__init__()
        self.inner_layer1=[]
        self.inner_layer2=[]
        self.out_layer1=[]
        self.out_layer2=[]
        self.upsp=[]
        self.ini=nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=1)
        # self.ini=rv.RepVGGplusBlock(out_channel, out_channel, padding=1)
        self.col_half=nn.Conv2d(in_channels=out_channel*2, out_channels=out_channel, kernel_size=1)

        for in_channel in in_channel_list:
            self.inner_layer1.append(nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=1))
            self.out_layer1.append(rv.RepVGGplusBlock(out_channel, out_channel, padding=1))
            self.inner_layer2.append(nn.Conv2d(in_channels=out_channel*2, out_channels=out_channel, kernel_size=1))
            self.out_layer2.append(rv.RepVGGplusBlock(out_channel, out_channel, padding=1))
            # self.upsp.append(rv.RepVGGplusBlock(out_channel, out_channel, padding=1))
            self.upsp.append(nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=1, stride=2))

    def forward(self, x):
        head_output=[]
        panhead_output=[]
        a = self.inner_layer1[-1]
        b = x[-1]
        current_inner=self.inner_layer1[-1](x[-1])
        head_output.append(self.out_layer1[-1](current_inner))

        for i in range(len(x)-2,-1,-1):
            pre_inner=current_inner
            current_inner=self.inner_layer1[i](x[i])
            size=current_inner.shape[2:]
            top_down=F.interpolate(pre_inner, size=size)
            cat_pre2current=torch.cat([top_down, current_inner], dim=1)
            col_half=self.col_half(cat_pre2current)
            head_output.append(self.out_layer1[i](col_half))

        after=self.ini(head_output[len(x)-1])
        panhead_output.append(self.out_layer2[len(x)-1](after))
        after=self.upsp[len(x)-1](after)
        after=self.Judge(after)

        for i in range(len(x)-2,-1,-1):
            after=torch.cat([head_output[i], after], dim=1)
            after=self.inner_layer2[i](after)
            panhead_output.append(self.out_layer2[i](after))
            after = self.Judge(after)
            after=self.upsp[i](after)


        final=panhead_output[len(x)-1].view(*panhead_output[len(x)-1].shape[:-2], -1)

        for i in range(len(x)-1):
            temp=panhead_output[i].view(*panhead_output[i].shape[:-2], -1)
            final=torch.cat([final, temp], dim=2)

        return final

    def Judge(self, x):
        if x.size()[3] % 2 != 0:
            x = x[:, :, :, :-1]
        if x.size()[2] % 2 != 0:
            x = x[:, :, :-1, :]
        return x

