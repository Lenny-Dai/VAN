import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import RepVGG as rv

class FPN(nn.Module):
    def __init__(self, in_channel_list, out_channel):
        super(FPN, self).__init__()
        self.inner_layer=[]
        self.out_layer=[]
        self.col_half=nn.Conv2d(in_channels=out_channel*2, out_channels=out_channel, kernel_size=1)
        for in_channel in in_channel_list:
            self.inner_layer.append(nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=1))
            self.out_layer.append(rv.RepVGGplusBlock(out_channel, out_channel, padding=1))

    def forward(self, x):
        head_output=[]
        current_inner=self.inner_layer[-1](x[-1])
        head_output.append(self.out_layer[-1](current_inner))

        for i in range(len(x)-2,-1,-1):
            pre_inner=current_inner
            current_inner=self.inner_layer[i](x[i])
            size=current_inner.shape[2:]
            top_down=F.interpolate(pre_inner, size=size)
            cat_pre2current=torch.cat([top_down, current_inner], dim=1)
            col_half=self.col_half(cat_pre2current)
            head_output.append(self.out_layer[i](col_half))

        final=head_output[len(x)-1].view(*head_output[len(x)-1].shape[:-2], -1)

        for i in range(len(x)-1):
            temp=head_output[i].view(*head_output[i].shape[:-2], -1)
            final=torch.cat([final, temp], dim=3)

        return final

