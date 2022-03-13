import math
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F


#-------------------------------------------------#
#   MISH激活函数
#-------------------------------------------------#
class Mish(nn.Module):
    def __init__(self):
        super(Mish, self).__init__()

    def forward(self, x):
        return x * torch.tanh(F.softplus(x))

#---------------------------------------------------#
#   卷积块 -> 卷积 + 标准化 + 激活函数
#   Conv2d + BatchNormalization + Mish
#---------------------------------------------------#
class BasicConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super(BasicConv, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, kernel_size//2, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = Mish()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x

#---------------------------------------------------#
#   CSPdarknet的结构块的组成部分
#   内部堆叠的残差块
#---------------------------------------------------#
class Resblock(nn.Module):
    def __init__(self, channels, hidden_channels=None):
        '''
        1x1基本卷积+3x3基本卷积
        :param channels: 输出channel
        :param hidden_channels: 第一个1x1卷积输出的隐藏层,
        然后经过3x3卷积会输出对应的out_channel
        '''
        super(Resblock, self).__init__()

        if hidden_channels is None:
            hidden_channels = channels

        self.block = nn.Sequential(
            BasicConv(channels, hidden_channels, 1),
            BasicConv(hidden_channels, channels, 3)
        )

    def forward(self, x):
        return x + self.block(x)

#--------------------------------------------------------------------#
#   CSPdarknet的结构块
#   首先利用ZeroPadding2D和一个步长为2x2的卷积块进行高和宽的压缩
#   然后建立一个大的残差边shortconv、这个大残差边绕过了很多的残差结构
#   主干部分会对num_blocks进行循环，循环内部是残差结构。
#   对于整个CSPdarknet的结构块，就是一个大残差块+内部多个小残差块
#--------------------------------------------------------------------#
class Resblock_body(nn.Module):
    def __init__(self, in_channels, out_channels, num_blocks, first):
        super(Resblock_body, self).__init__()
        #----------------------------------------------------------------#
        #   利用一个步长为2x2的卷积块进行高和宽的压缩
        #----------------------------------------------------------------#
        self.downsample_conv = BasicConv(in_channels, out_channels, 3, stride=2)

        if first:
            '''
            因为第一个卷积输出为32channel,此时特征图数量较少
            而经过3x3缩减特征图尺寸后,out_channel变成64
            不适合做输入直接通过1x1卷积提取成两半 再分别卷积
            所以第一个residual_body采用的方式是short和残差模块都输出out_channel(64)的形式,
            然后堆叠成2*out_channel(128),再通过1x1卷积降维成最终输出64
            '''
            #--------------------------------------------------------------------------#
            #   然后建立一个大的残差边self.split_conv0、这个大残差边绕过了很多的残差结构
            #--------------------------------------------------------------------------#
            self.split_conv0 = BasicConv(out_channels, out_channels, 1)

            #----------------------------------------------------------------#
            #   主干部分会对num_blocks进行循环，循环内部是残差结构。
            #----------------------------------------------------------------#
            self.split_conv1 = BasicConv(out_channels, out_channels, 1)  
            self.blocks_conv = nn.Sequential(
                # 第一个residual模块也只重复一次
                Resblock(channels=out_channels, hidden_channels=out_channels//2),
                BasicConv(out_channels, out_channels, 1)
            )

            self.concat_conv = BasicConv(out_channels*2, out_channels, 1)
        else:
            #--------------------------------------------------------------------------#
            #   然后建立一个大的残差边self.split_conv0、这个大残差边绕过了很多的残差结构
            #--------------------------------------------------------------------------#
            self.split_conv0 = BasicConv(out_channels, out_channels//2, 1)

            #----------------------------------------------------------------#
            #   主干部分会对num_blocks进行循环，循环内部是残差结构。
            #----------------------------------------------------------------#
            self.split_conv1 = BasicConv(out_channels, out_channels//2, 1)
            self.blocks_conv = nn.Sequential(
                # num_blocks表示残差块的个数
                # 这里的残差块并不像CSPNet或者densenet中的相互堆叠,越往后面channel越大
                # 这里是越往后面,每层的channel尺度不变
                *[Resblock(out_channels//2) for _ in range(num_blocks)],
                # 串联完num_blocks个residual后,再来一个1x1卷积融合特征
                BasicConv(out_channels//2, out_channels//2, 1)
            )

            # 最后输出之前再来个1x1卷积,将concatenate的两层融合
            self.concat_conv = BasicConv(out_channels, out_channels, 1)

    def forward(self, x):
        x = self.downsample_conv(x)

        x0 = self.split_conv0(x)

        x1 = self.split_conv1(x)
        x1 = self.blocks_conv(x1)

        #------------------------------------#
        #   将大残差边再堆叠回来
        #------------------------------------#
        x = torch.cat([x1, x0], dim=1)
        #------------------------------------#
        #   最后对通道数进行整合
        #------------------------------------#
        x = self.concat_conv(x)

        return x

#---------------------------------------------------#
#   CSPdarknet53 的主体部分
#   输入为一张416x416x3的图片
#   输出为三个有效特征层
#---------------------------------------------------#
class CSPDarkNet(nn.Module):
    def __init__(self, layers):
        super(CSPDarkNet, self).__init__()
        self.inplanes = 32
        # 416,416,3 -> 416,416,32
        self.conv1 = BasicConv(3, self.inplanes, kernel_size=3, stride=1)
        self.feature_channels = [64, 128, 256, 512, 1024]

        self.stages = nn.ModuleList([
            # 416,416,32 -> 208,208,64
            # 仅第一个输出前的的1x1卷积是降维 128->64,其他输出前的1x1卷积都是直接同维度变换
            Resblock_body(self.inplanes, self.feature_channels[0], layers[0], first=True),
            # 208,208,64 -> 104,104,128
            Resblock_body(self.feature_channels[0], self.feature_channels[1], layers[1], first=False),
            # 104,104,128 -> 52,52,256
            Resblock_body(self.feature_channels[1], self.feature_channels[2], layers[2], first=False),
            # 52,52,256 -> 26,26,512
            Resblock_body(self.feature_channels[2], self.feature_channels[3], layers[3], first=False),
            # 26,26,512 -> 13,13,1024
            Resblock_body(self.feature_channels[3], self.feature_channels[4], layers[4], first=False)
        ])

        self.num_features = 1
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


    def forward(self, x):
        x = self.conv1(x)

        x = self.stages[0](x)
        x = self.stages[1](x)
        out3 = self.stages[2](x)
        out4 = self.stages[3](out3)
        out5 = self.stages[4](out4)

        return out3, out4, out5
    
def darknet53(pretrained):
    model = CSPDarkNet([1, 2, 8, 8, 4])
    if pretrained:
        model.load_state_dict(torch.load("model_data/CSPdarknet53_backbone_weights.pth"))
    return model


if __name__ == '__main__':
    from torch.utils.tensorboard import SummaryWriter

    net=darknet53(pretrained=False)
    data=torch.randn(1,3,416,416)
    out=net(data)
    with SummaryWriter(log_dir='./log', comment='cspdarknet') as writer:
        writer.add_graph(net, data)