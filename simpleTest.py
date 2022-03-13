#!/usr/bin/env python
#-*- coding:utf-8 _*-
__author__ = 'LJjia'
# *******************************************************************
#     Filename @  simpleTest.py
#       Author @  Jia Liangjun
#  Create date @  2022/03/08 12:11
#        Email @  LJjiahf@163.com
#  Description @  
# ********************************************************************


import torch
from PIL import Image
from utils.utils import cvtColor, preprocess_input
import cv2
import numpy as np


class Rand(object):
    def __init__(self):
        self.mosic=True
    def rand(self,a=0.,b=1.):
        return np.random.rand()*(b-a) + a


def get_random_data_with_Mosaic(self, annotation_line, input_shape, max_boxes=100, hue=.1, sat=1.5, val=1.5):
    '''
    将4张图融合成1张图
    :param annotation_line:
    :param input_shape:
    :param max_boxes:
    :param hue:
    :param sat:
    :param val:
    :return:
    '''
    h, w = input_shape
    # 选取中心点
    min_offset_x = self.rand(0.25, 0.75)
    min_offset_y = self.rand(0.25, 0.75)

    # 应该是生成每张图片缩放后的宽高
    nws = [int(w * self.rand(0.4, 1)), int(w * self.rand(0.4, 1)), int(w * self.rand(0.4, 1)),
           int(w * self.rand(0.4, 1))]
    nhs = [int(h * self.rand(0.4, 1)), int(h * self.rand(0.4, 1)), int(h * self.rand(0.4, 1)),
           int(h * self.rand(0.4, 1))]

    # 图片放置的左上角xy坐标 顺序为
    # 1 4
    # 2 3
    # 注意palce_x和place_y都是像素单位的坐标
    place_x = [int(w * min_offset_x) - nws[0], int(w * min_offset_x) - nws[1], int(w * min_offset_x),
               int(w * min_offset_x)]
    place_y = [int(h * min_offset_y) - nhs[0], int(h * min_offset_y), int(h * min_offset_y),
               int(h * min_offset_y) - nhs[3]]

    image_datas = []
    box_datas = []
    index = 0
    for line in annotation_line:
        # 每一行进行分割
        line_content = line.split()
        # 打开图片
        image = Image.open(line_content[0])
        image = cvtColor(image)

        # 图片的大小
        iw, ih = image.size
        # 保存框的位置 box似乎是个5维,最后一维为类别
        box = np.array([np.array(list(map(int, box.split(',')))) for box in line_content[1:]])

        # 是否翻转图片
        flip = self.rand() < .5
        if flip and len(box) > 0:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            box[:, [0, 2]] = iw - box[:, [2, 0]]

        nw = nws[index]
        nh = nhs[index]
        image = image.resize((nw, nh), Image.BICUBIC)

        # 将图片进行放置，分别对应四张分割图片的位置
        dx = place_x[index]
        dy = place_y[index]
        new_image = Image.new('RGB', (w, h), (128, 128, 128))
        new_image.paste(image, (dx, dy))
        image_data = np.array(new_image)

        index = index + 1
        box_data = []
        # 对box进行重新处理
        if len(box) > 0:
            np.random.shuffle(box)
            box[:, [0, 2]] = box[:, [0, 2]] * nw / iw + dx
            box[:, [1, 3]] = box[:, [1, 3]] * nh / ih + dy
            box[:, 0:2][box[:, 0:2] < 0] = 0
            box[:, 2][box[:, 2] > w] = w
            box[:, 3][box[:, 3] > h] = h
            box_w = box[:, 2] - box[:, 0]
            box_h = box[:, 3] - box[:, 1]
            # 删除框框宽高<1的,以及框框部分和其他图片重合的
            box = box[np.logical_and(box_w > 1, box_h > 1)]
            box_data = np.zeros((len(box), 5))
            box_data[:len(box)] = box
        # 每个image_data或box_data中存一个只有1/4的图片和位置信息?
        image_datas.append(image_data)
        box_datas.append(box_data)

    # 将图片分割，放在一起
    cutx = int(w * min_offset_x)
    cuty = int(h * min_offset_y)

    # 这样申请出来的new_image竟然是float64数据,奇葩
    new_image = np.zeros([h, w, 3],dtype=np.uint8)
    # 注意,因为从Image变为array的过程中,wh发生变换
    # 如Image中是(w,h)
    # array中则是(h,w,channel)
    # 所以下面写的奇怪,是先取y轴,再取x轴
    new_image[:cuty, :cutx, :] = image_datas[0][:cuty, :cutx, :]
    new_image[cuty:, :cutx, :] = image_datas[1][cuty:, :cutx, :]
    new_image[cuty:, cutx:, :] = image_datas[2][cuty:, cutx:, :]
    new_image[:cuty, cutx:, :] = image_datas[3][:cuty, cutx:, :]
    tmp_img = Image.fromarray(new_image)
    tmp_img.show()
    # 进行色域变换
    hue = self.rand(-hue, hue)
    sat = self.rand(1, sat) if self.rand() < .5 else 1 / self.rand(1, sat)
    val = self.rand(1, val) if self.rand() < .5 else 1 / self.rand(1, val)
    x = cv2.cvtColor(np.array(new_image / 255, np.float32), cv2.COLOR_RGB2HSV)
    x[..., 0] += hue * 360
    x[..., 0][x[..., 0] > 1] -= 1
    x[..., 0][x[..., 0] < 0] += 1
    x[..., 1] *= sat
    x[..., 2] *= val
    x[x[:, :, 0] > 360, 0] = 360
    x[:, :, 1:][x[:, :, 1:] > 1] = 1
    x[x < 0] = 0
    new_image = cv2.cvtColor(x, cv2.COLOR_HSV2RGB) * 255

    # 对框进行进一步的处理
    new_boxes = self.merge_bboxes(box_datas, cutx, cuty)

    return new_image, new_boxes

if __name__ == '__main__':
    w,h=416,416
    dx,dy=-100,-100
    image=Image.open('img/car.jpg')
    new_image = Image.new('RGB', (w, h), (128, 128, 128))
    new_image.paste(image, (dx, dy))
    new_image.show()