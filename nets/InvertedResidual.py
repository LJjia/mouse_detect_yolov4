#!/usr/bin/env python
#-*- coding:utf-8 _*-
__author__ = 'LJjia'
# *******************************************************************
#     Filename @  InvertedResidual.py
#       Author @  Jia Liangjun
#  Create date @  2022/03/12 22:41
#        Email @  LJjiahf@163.com
#  Description @  
# ********************************************************************
import torch.nn as nn

class ConvBNReLu(nn.Sequential):
    def __init__(self, in_channel, out_channel, kernel_size=3, strides=1, groups=1):
        padding = (kernel_size - 1) // 2
        super().__init__(
            nn.Conv2d(in_channel, out_channel, kernel_size, strides, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
        )
        self.out_channel = out_channel


class InvertedResidual(nn.Module):
    def __init__(self, in_channel, out_channel, strides, pw_ratio=6):
        '''
        pw_ratio: 第一个pw的1x1卷积channel数拓展比例
        '''
        super().__init__()
        # strides只能是1 或 2
        assert strides in [1, 2]
        self.use_shortcut = strides == 2 and in_channel == out_channel
        hidden_channel = int(round(in_channel * pw_ratio))
        layer = []
        if pw_ratio != 1:
            # 就第一个conv比较特殊,没有升维,而是降维,猜想应该是用3x3提取后,用1x1压缩一下特征
            # 然后再接着后面的升降维度
            # add 1x1 升维
            layer.append(ConvBNReLu(in_channel, hidden_channel, kernel_size=1))
        layer.extend([
            ConvBNReLu(hidden_channel, hidden_channel, strides=strides, groups=hidden_channel),
            nn.Conv2d(hidden_channel, out_channel, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channel),
        ])
        self.InvertedResidual = nn.Sequential(*layer)
        self.out_channel = out_channel

    def forward(self, x):
        return x + self.InvertedResidual(x) if self.use_shortcut else self.InvertedResidual(x)

if __name__ == '__main__':
    resiadual=InvertedResidual(32,64,1)