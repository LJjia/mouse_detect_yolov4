from collections import OrderedDict

import torch
import torch.nn as nn

from nets.CSPdarknet import darknet53
from nets.ghostnet import GhostNet
from nets.shuffnetv2 import shufflenet_v2_x1_0
from nets.InvertedResidual import InvertedResidual
from enum import Enum

def conv2d(filter_in, filter_out, kernel_size, stride=1):
    '''
    conv+bn+LeakRelu 模块
    :param filter_in: 输入channel
    :param filter_out: 输出channel
    :param kernel_size: kersize大小
    :param stride: 跨距默认1
    :return:
    '''
    pad = (kernel_size - 1) // 2 if kernel_size else 0
    return nn.Sequential(OrderedDict([
        ("conv", nn.Conv2d(filter_in, filter_out, kernel_size=kernel_size, stride=stride, padding=pad, bias=False)),
        ("bn", nn.BatchNorm2d(filter_out)),
        ("relu", nn.LeakyReLU(0.1)),
    ]))

#---------------------------------------------------#
#   SPP结构，利用不同大小的池化核进行池化
#   池化后堆叠
#---------------------------------------------------#
class SpatialPyramidPooling(nn.Module):
    def __init__(self, pool_sizes=[5, 9, 13]):
        super(SpatialPyramidPooling, self).__init__()

        # kersize=pool_size
        # stride=1
        # pool=pool_size//2
        # 这么计算下来,对于13x13的特征图,每个pool_sizes的输出都是13x13的特征图...
        self.maxpools = nn.ModuleList([nn.MaxPool2d(pool_size, 1, pool_size//2) for pool_size in pool_sizes])

    def forward(self, x):
        # 输出先是pool=13的特征图结果,然后pool=9,pool=5,pool=1,堆叠
        features = [maxpool(x) for maxpool in self.maxpools[::-1]]
        features = torch.cat(features + [x], dim=1)

        return features

#---------------------------------------------------#
#   卷积 + 上采样
#---------------------------------------------------#
class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        '''
        卷积+上采样, 输出特征图恢复成2倍长宽大小
        :param in_channels: 输入通道数
        :param out_channels: 输出通道数
        '''
        super(Upsample, self).__init__()
        in_channels=int(in_channels)
        out_channels=int(out_channels)
        self.upsample = nn.Sequential(
            conv2d(in_channels, out_channels, 1),
            nn.Upsample(scale_factor=2, mode='nearest')
        )

    def forward(self, x,):
        x = self.upsample(x)
        return x

#---------------------------------------------------#
#   三次卷积块 其实是一个倒残差模块,中间维度变换已经被内定6倍了
#---------------------------------------------------#
def make_three_conv(in_filters,out_filters):

    '''
    三层卷积,分别为1x1压缩,3x3特征提取,1x1还原
    三次卷积中每个都是conv_bn_relu
    :param filters_list: 两个元素组成的list filters_list[0]表示输出n,filters_list[1]表示中间层深度
    :param in_filters: 输入深度
    :return:
    '''
    in_filters=int(in_filters)
    out_filters=int(out_filters)
    InvertedResThree=nn.Sequential(
        InvertedResidual(in_filters, out_filters,1)
    )
    return InvertedResThree
    # m = nn.Sequential(
    #     conv2d(in_filters, filters_list[0], 1),
    #     conv2d(filters_list[0], filters_list[1], 3),
    #     conv2d(filters_list[1], filters_list[0], 1),
    # )
    # return m

#---------------------------------------------------#
#   五次卷积块,其实是两个倒残差模块的组合,中间维度变换已经被内定6倍了
#---------------------------------------------------#
def make_five_conv(in_filters,out_filters):
    '''
    1x1降维 3x3升维 1x1降维 3x3升维 1x1降维为输出
    :param filters_list: [0]:输出channel [1]:中间层升维的深度
    :param in_filters: 输入channel长度
    :return:
    '''
    in_filters = int(in_filters)
    out_filters = int(out_filters)
    InvertedResFive=nn.Sequential(
        InvertedResidual(in_filters, out_filters,1),
        InvertedResidual(out_filters, out_filters,1)
    )
    return InvertedResFive
    # m = nn.Sequential(
    #     conv2d(in_filters, filters_list[0], 1),
    #     conv2d(filters_list[0], filters_list[1], 3),
    #     conv2d(filters_list[1], filters_list[0], 1),
    #     conv2d(filters_list[0], filters_list[1], 3),
    #     conv2d(filters_list[1], filters_list[0], 1),
    # )
    # return m

#---------------------------------------------------#
#   最后获得yolov4的输出
#---------------------------------------------------#
def yolo_head(in_filters,out_filters,hid_ratio=2):
    '''
    3x3卷积加深通道+1x1卷积输出 75类别
    :param filters_list:两个元素组成的list filters_list[0]表示中间层深度,filters_list[1]表示输出
    :param in_filters:输入深度
    :return:
    '''
    head = nn.Sequential(
        conv2d(in_filters, in_filters*hid_ratio, 3),
        nn.Conv2d(in_filters*hid_ratio, out_filters, 1),
    )
    return head

#---------------------------------------------------#
#   yolo_body
#---------------------------------------------------#
class YoloBody(nn.Module):
    backbone_type = Enum("backbone_type", ('ghost', 'shufflenet', 'darknet53'))
    def __init__(self, anchors_mask, num_classes, net_name):
        '''

        :param anchors_mask: 先验框尺寸,这里仅用到每个尺寸的特征图一共有几个先验框
        :param num_classes: 种类个数如voc数据集为20
        :param pretrained:
        '''
        super(YoloBody, self).__init__()
        #---------------------------------------------------#   
        #   生成CSPdarknet53的主干模型
        #   获得三个有效特征层，他们的shape分别是：
        #   52,52,256
        #   26,26,512
        #   13,13,1024
        #---------------------------------------------------#
        # 默认预加载模型
        if net_name==YoloBody.backbone_type.ghost:
            self.backbone=GhostNet()
        elif net_name==YoloBody.backbone_type.shufflenet:
            self.backbone=shufflenet_v2_x1_0()
        elif net_name==YoloBody.backbone_type.darknet53:
            # 返回dark53主骨干网络,不带特征提取层
            self.backbone = darknet53()
        else:
            print('input correct net name .you input',net_name)
            raise ValueError("input incorrect net name")
        self.net_name = net_name
        self.backbone_fliter=self.backbone.layers_out_filters
        # 值由小到大,表示低层到高层的卷积核输出尺度
        # 例如ghost深度为40,112,160
        chn1,chn2,chn3=self.backbone_fliter
        # 160->80
        self.conv1      = make_three_conv(chn3,chn3/2)
        self.SPP        = SpatialPyramidPooling()
        # 320->160
        self.conv2      = make_three_conv(chn3*2,chn3)
        # 160->80
        self.upsample1          = Upsample(chn3,chn3/2)
        # 112->112 为了防止特征降的太多
        self.conv_for_P4        = conv2d(chn2,chn2,1)
        # 112+80->112
        self.make_five_conv1    = make_five_conv(chn2+chn3/2,chn2)
        # 112->56
        self.upsample2          = Upsample(chn2,chn2/2)
        # 40->40
        self.conv_for_P3        = conv2d(chn1,chn1,1)
        # 40+56->40
        self.make_five_conv2    = make_five_conv(chn1+chn2/2,chn1)

        # 52,52,40-> 3*(5+num_classes) = 3*(5+20) = 3*(4+1+20)=75
        self.yolo_head3         = yolo_head(chn1, len(anchors_mask[0]) * (5 + num_classes))
        # 40->80
        self.down_sample1       = conv2d(chn1,chn1*2,3,stride=2)
        # 80+112->112
        self.make_five_conv3    = make_five_conv(chn1*2+chn2,chn2)

        # 26,26,112->3*(5+num_classes) = 3*(5+20) = 3*(4+1+20)=75
        self.yolo_head2         = yolo_head(chn2, len(anchors_mask[1]) * (5 + num_classes))
        # 112->224
        self.down_sample2       = conv2d(chn2,chn2*2,3,stride=2)
        # 224+160->160
        self.make_five_conv4    = make_five_conv(chn2*2+chn3,chn3)

        # 13,13,160->3*(5+num_classes)=3*(5+20)=3*(4+1+20)=75
        self.yolo_head1         = yolo_head(chn3, len(anchors_mask[2]) * (5 + num_classes))


    def forward(self, x):
        #  backbone
        #   获得三个有效特征层，他们的shape分别是：
        #   x2:52,52,256
        #   x1:26,26,512
        #   x0:13,13,1024
        x2, x1, x0 = self.backbone(x)

        # conv1,三次卷积,输入1024,输出512
        P5 = self.conv1(x0)
        # SPP多尺度maxpool,特征图尺度不变,深度变为4倍,变成[13,13,2048]
        P5 = self.SPP(P5)
        # [13,13,2048]->[13,13,512]
        P5 = self.conv2(P5)
        # [13,13,512]->[26,26,256]
        P5_upsample = self.upsample1(P5)

        # [26,26,512]->[26,26,256]
        P4 = self.conv_for_P4(x1)
        # 融合成[26,26,512]
        P4 = torch.cat([P4,P5_upsample],axis=1)
        # [26,26,512]->[26,26,256]
        P4 = self.make_five_conv1(P4)

        # [26,26,256]->[52,52,128]
        P4_upsample = self.upsample2(P4)
        # [52,52,256]->[52,52,128]
        P3 = self.conv_for_P3(x2)
        # [52,52,256]
        P3 = torch.cat([P3,P4_upsample],axis=1)
        # [52,52,256]->[52,52,128]
        P3 = self.make_five_conv2(P3)

        # [52,52,128]->[26,26,256]
        P3_downsample = self.down_sample1(P3)
        # [26,26,512]
        P4 = torch.cat([P3_downsample,P4],axis=1)
        # [26,26,512]->[26,26,256]
        P4 = self.make_five_conv3(P4)

        # [26,26,256]->[13,13,512]
        P4_downsample = self.down_sample2(P4)
        # [13,13,1024]
        P5 = torch.cat([P4_downsample,P5],axis=1)
        # [13,13,1024]->[13,13,512]
        P5 = self.make_five_conv4(P5)

        #---------------------------------------------------#
        #   第三个特征层 P3:[52,52,128]
        #   y3=(batch_size,75,52,52)
        #---------------------------------------------------#
        out2 = self.yolo_head3(P3)
        #---------------------------------------------------#
        #   第二个特征层 P4:[26,26,256]
        #   y2=(batch_size,75,26,26)
        #---------------------------------------------------#
        out1 = self.yolo_head2(P4)
        #---------------------------------------------------#
        #   第一个特征层 P5:[13,13,512]
        #   y1=(batch_size,75,13,13)
        #---------------------------------------------------#
        out0 = self.yolo_head1(P5)

        return out0, out1, out2

    def load_backbone(self):
        if self.net_name==YoloBody.backbone_type.ghost:
            self.backbone.load_model("model_data/change_official_ghost.pth")
        if self.net_name==YoloBody.backbone_type.shufflenet:
            self.backbone.load_model("model_data/shufflenetv2_x1.pth")
        elif self.net_name==YoloBody.backbone_type.darknet53:
            self.backbone.load_state_dict(torch.load("model_data/darknet53_backbone_weights.pth"))
        else:
            print("net name %s net weight"%self.net_name)



if __name__ == '__main__':
    anchors_mask=[[6, 7, 8], [3, 4, 5], [0, 1, 2]]
    num_classes=10
    model = YoloBody(anchors_mask, num_classes, net_name=YoloBody.backbone_type.ghost)
    print('run end')