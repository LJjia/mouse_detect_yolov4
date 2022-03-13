#!/usr/bin/env python
#-*- coding:utf-8 _*-
__author__ = 'LJjia'
# *******************************************************************
#     Filename @  shuffnetv2.py
#       Author @  Jia Liangjun
#  Create date @  2022/03/07 16:13
#        Email @  LJjiahf@163.com
#  Description @  shuffnetv2 作为backbone
# ********************************************************************

from typing import Callable, Any, List
import torch
import torch.nn as nn
from torch import Tensor


__all__ = ["ShuffleNetV2", "shufflenet_v2_x0_5", "shufflenet_v2_x1_0"]


def channel_shuffle(x: Tensor, groups: int) -> Tensor:
    batchsize, num_channels, height, width = x.size()
    channels_per_group = num_channels // groups

    # reshape
    x = x.view(batchsize, groups, channels_per_group, height, width)

    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width)

    return x


class InvertedResidual(nn.Module):
    def __init__(self, inp: int, oup: int, stride: int) -> None:
        super().__init__()

        if not (1 <= stride <= 3):
            raise ValueError("illegal stride value")
        self.stride = stride

        branch_features = oup // 2
        assert (self.stride != 1) or (inp == branch_features << 1)

        if self.stride > 1:
            self.branch1 = nn.Sequential(
                self.depthwise_conv(inp, inp, kernel_size=3, stride=self.stride, padding=1),
                nn.BatchNorm2d(inp),
                nn.Conv2d(inp, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(branch_features),
                nn.ReLU(inplace=True),
            )
        else:
            self.branch1 = nn.Sequential()

        self.branch2 = nn.Sequential(
            nn.Conv2d(
                inp if (self.stride > 1) else branch_features,
                branch_features,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),
            self.depthwise_conv(branch_features, branch_features, kernel_size=3, stride=self.stride, padding=1),
            nn.BatchNorm2d(branch_features),
            nn.Conv2d(branch_features, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),
        )

    @staticmethod
    def depthwise_conv(
        i: int, o: int, kernel_size: int, stride: int = 1, padding: int = 0, bias: bool = False
    ) -> nn.Conv2d:
        return nn.Conv2d(i, o, kernel_size, stride, padding, bias=bias, groups=i)

    def forward(self, x: Tensor) -> Tensor:
        if self.stride == 1:
            x1, x2 = x.chunk(2, dim=1)
            out = torch.cat((x1, self.branch2(x2)), dim=1)
        else:
            out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)

        out = channel_shuffle(out, 2)

        return out


class ShuffleNetV2(nn.Module):
    def __init__(
        self,
        stages_repeats: List[int],
        stages_out_channels: List[int],
        num_classes: int = 1000,
        inverted_residual: Callable[..., nn.Module] = InvertedResidual,
    ) -> None:
        super().__init__()

        if len(stages_repeats) != 3:
            raise ValueError("expected stages_repeats as list of 3 positive ints")
        if len(stages_out_channels) != 5:
            raise ValueError("expected stages_out_channels as list of 5 positive ints")
        self._stage_out_channels = stages_out_channels

        input_channels = 3
        output_channels = self._stage_out_channels[0]
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 3, 2, 1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        )
        input_channels = output_channels

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Static annotations for mypy
        self.stage2: nn.Sequential
        self.stage3: nn.Sequential
        self.stage4: nn.Sequential
        stage_names = [f"stage{i}" for i in [2, 3, 4]]
        for name, repeats, output_channels in zip(stage_names, stages_repeats, self._stage_out_channels[1:]):
            seq = [inverted_residual(input_channels, output_channels, 2)]
            for i in range(repeats - 1):
                seq.append(inverted_residual(output_channels, output_channels, 1))
            setattr(self, name, nn.Sequential(*seq))
            input_channels = output_channels

        output_channels = self._stage_out_channels[-1]
        # self.conv5 = nn.Sequential(
        #     nn.Conv2d(input_channels, output_channels, 1, 1, 0, bias=False),
        #     nn.BatchNorm2d(output_channels),
        #     nn.ReLU(inplace=True),
        # )
        #
        # self.fc = nn.Linear(output_channels, num_classes)
        self.layers_out_filters = [116, 232, 464]
        self.weight_init()

    def forward(self, x: Tensor):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.stage2(x)
        feat1 = x
        x = self.stage3(x)
        feat2 = x
        x = self.stage4(x)
        feat3 = x
        # x = self.conv5(x)
        # x = x.mean([2, 3])  # globalpool
        # x = self.fc(x)

        # 1x 输出尺度
        # 116x52x52
        # 232x26x26
        # 464x13x13
        return (feat1,feat2,feat3)

    def weight_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # 一种正态分布
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.num_batches_tracked)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def load_model(self,path):
        model_dict = self.state_dict()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # 加载模型到device上
        pretrained_dict = torch.load(path, map_location=device)
        # 预训练模型和本地模型shape相等则加载权重
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if
                           (k in model_dict and np.shape(model_dict[k]) == np.shape(v))}
        # 查看未加载的权重
        # for k in model_dict.keys():
        #     if k not in pretrained_dict:
        #         print("local model layer %s not load!" % k, )
        # print("from %s load all model weight",path)
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)


def _shufflenetv2(arch: str, pretrained: bool, progress: bool, *args: Any, **kwargs: Any) -> ShuffleNetV2:
    model = ShuffleNetV2(*args, **kwargs)

    if pretrained:
        state_dict = torch.load('shufflenetv2_x1.pth')
        print('pretrain dict')
        for k in state_dict.keys():
            print(k)
        model.load_state_dict(state_dict)

    return model


def shufflenet_v2_x0_5(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ShuffleNetV2:
    """
    Constructs a ShuffleNetV2 with 0.5x output channels, as described in
    `"ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design"
    <https://arxiv.org/abs/1807.11164>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _shufflenetv2("shufflenetv2_x0.5", pretrained, progress, [4, 8, 4], [24, 48, 96, 192, 1024], **kwargs)


def shufflenet_v2_x1_0(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ShuffleNetV2:
    """
    Constructs a ShuffleNetV2 with 1.0x output channels, as described in
    `"ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design"
    <https://arxiv.org/abs/1807.11164>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _shufflenetv2("shufflenetv2_x1.0", pretrained, progress, [4, 8, 4], [24, 116, 232, 464, 1024], **kwargs)


if __name__ == '__main__':
    from nets.backbone_test import *
    net=shufflenet_v2_x1_0(pretrained=False)
    # show_dict_keys(net.state_dict())
    # print('*'*50)
    # show_dict_keys(torch.load('shufflenetv2_x1.pth'))

    # net_test(net,'model_data/shufflenetv2_x1.pth','../img/cat_lab285.jpg',net_type='feature')