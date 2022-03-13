from random import sample, shuffle

import cv2
import numpy as np
from PIL import Image
from torch.utils.data.dataset import Dataset
import torch
from utils.utils import cvtColor, preprocess_input


class YoloDataset(Dataset):
    def __init__(self, annotation_lines, input_shape, num_classes, mosaic, train):
        '''

        :param annotation_lines: 图片path列表
        :param input_shape: [416,416]
        :param num_classes: 20
        :param mosaic: 是否mosaic
        :param train: 是否训练
        '''
        super(YoloDataset, self).__init__()
        self.annotation_lines   = annotation_lines
        self.input_shape        = input_shape
        self.num_classes        = num_classes
        self.length             = len(self.annotation_lines)
        self.mosaic             = mosaic
        self.train              = train
        self.mosaic_ratio=0.7

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        index       = index % self.length
        #---------------------------------------------------#
        #   训练时进行数据的随机增强
        #   验证时不进行数据的随机增强
        #---------------------------------------------------#
        if self.mosaic:
            if self.rand() < self.mosaic_ratio:
                # 先随机取出三个图片,然后加上当前的图片
                lines = sample(self.annotation_lines, 3)
                lines.append(self.annotation_lines[index])
                shuffle(lines)
                image, box  = self.get_random_data_with_Mosaic(lines, self.input_shape)
            else:
                image, box  = self.get_random_data(self.annotation_lines[index], self.input_shape, random = self.train)
        else:
            image, box      = self.get_random_data(self.annotation_lines[index], self.input_shape, random = self.train)
        image       = np.transpose(preprocess_input(np.array(image, dtype=np.float32)), (2, 0, 1))
        box         = np.array(box, dtype=np.float32)
        if len(box) != 0:
            box[:, [0, 2]] = box[:, [0, 2]] / self.input_shape[1]
            box[:, [1, 3]] = box[:, [1, 3]] / self.input_shape[0]

            box[:, 2:4] = box[:, 2:4] - box[:, 0:2]
            box[:, 0:2] = box[:, 0:2] + box[:, 2:4] / 2
        return image, box

    def rand(self, a=0., b=1.):
        # 使用torch的随机数加载
        with torch.no_grad():
            ret=torch.rand(1).item()*(b-a) + a
        return ret
        # 这里不使用np的随机数,因为多线程加载时,这些随机数每个线程可能相同
        # return np.random.rand()*(b-a) + a

    def get_random_data(self, annotation_line, input_shape, jitter=.3, hue=.1, sat=1.5, val=1.5, random=True):
        line    = annotation_line.split()
        #------------------------------#
        #   读取图像并转换成RGB图像
        #------------------------------#
        image   = Image.open(line[0])
        image   = cvtColor(image)
        #------------------------------#
        #   获得图像的高宽与目标高宽
        #------------------------------#
        iw, ih  = image.size
        h, w    = input_shape
        #------------------------------#
        #   获得预测框
        #------------------------------#
        box     = np.array([np.array(list(map(int,box.split(',')))) for box in line[1:]])
        # 随机取灰条或者黑条
        ratio=self.rand()
        if ratio<0.5:
            color=np.array([0,0,0])
            color=color+int(self.rand()*50)
            color = tuple(color)
        elif ratio<1:
            color = np.array([128,128,128])
            color = color + int(self.rand() * 50)
            color = tuple(color)
        else:
            color = (0, 0, 0)
        if not random:
            scale = min(w/iw, h/ih)
            nw = int(iw*scale)
            nh = int(ih*scale)
            dx = (w-nw)//2
            dy = (h-nh)//2

            #---------------------------------#
            #   将图像多余的部分加上灰条
            #---------------------------------#
            image       = image.resize((nw,nh), Image.BICUBIC)
            new_image   = Image.new('RGB', (w,h), color)
            new_image.paste(image, (dx, dy))
            image_data  = np.array(new_image, np.float32)

            #---------------------------------#
            #   对真实框进行调整
            #---------------------------------#
            if len(box)>0:
                np.random.shuffle(box)
                box[:, [0,2]] = box[:, [0,2]]*nw/iw + dx
                box[:, [1,3]] = box[:, [1,3]]*nh/ih + dy
                box[:, 0:2][box[:, 0:2]<0] = 0
                box[:, 2][box[:, 2]>w] = w
                box[:, 3][box[:, 3]>h] = h
                box_w = box[:, 2] - box[:, 0]
                box_h = box[:, 3] - box[:, 1]
                box = box[np.logical_and(box_w>1, box_h>1)] # discard invalid box

            return image_data, box
                
        #------------------------------------------#
        #   对图像进行缩放并且进行长和宽的扭曲
        #------------------------------------------#
        new_ar = w/h * self.rand(1-jitter,1+jitter) / self.rand(1-jitter,1+jitter)
        scale = self.rand(.25, 2)
        if new_ar < 1:
            nh = int(scale*h)
            nw = int(nh*new_ar)
        else:
            nw = int(scale*w)
            nh = int(nw/new_ar)
        # 这么处理是有可能处理出大于图像边界416,416的图片的,但是后面会对框框进行处理
        image = image.resize((nw,nh), Image.BICUBIC)

        #------------------------------------------#
        #   将图像多余的部分加上灰条
        #------------------------------------------#
        dx = int(self.rand(0, w-nw))
        dy = int(self.rand(0, h-nh))
        # 也有可能缩放后的图片比较大,直接粘贴到了图像外围的左上角
        # 没关系不会报错的
        new_image = Image.new('RGB', (w,h), color)
        new_image.paste(image, (dx, dy))
        image = new_image

        #------------------------------------------#
        #   翻转图像
        #------------------------------------------#
        flip = self.rand()<.5
        if flip: image = image.transpose(Image.FLIP_LEFT_RIGHT)

        #------------------------------------------#
        #   色域扭曲
        #------------------------------------------#
        hue = self.rand(-hue, hue)
        sat = self.rand(1, sat) if self.rand()<.5 else 1/self.rand(1, sat)
        val = self.rand(1, val) if self.rand()<.5 else 1/self.rand(1, val)
        x = cv2.cvtColor(np.array(image,np.float32)/255, cv2.COLOR_RGB2HSV)
        x[..., 0] += hue*360
        x[..., 0][x[..., 0]>1] -= 1
        x[..., 0][x[..., 0]<0] += 1
        x[..., 1] *= sat
        x[..., 2] *= val
        x[x[:,:, 0]>360, 0] = 360
        x[:, :, 1:][x[:, :, 1:]>1] = 1
        x[x<0] = 0
        image_data = cv2.cvtColor(x, cv2.COLOR_HSV2RGB)*255

        #---------------------------------#
        #   对真实框进行调整
        #---------------------------------#
        if len(box)>0:
            np.random.shuffle(box)
            box[:, [0,2]] = box[:, [0,2]]*nw/iw + dx
            box[:, [1,3]] = box[:, [1,3]]*nh/ih + dy
            if flip: box[:, [0,2]] = w - box[:, [2,0]]
            box[:, 0:2][box[:, 0:2]<0] = 0
            box[:, 2][box[:, 2]>w] = w
            box[:, 3][box[:, 3]>h] = h
            box_w = box[:, 2] - box[:, 0]
            box_h = box[:, 3] - box[:, 1]
            box = box[np.logical_and(box_w>1, box_h>1)] 
        
        return image_data, box
    
    def merge_bboxes(self, bboxes, cutx, cuty,threshold):
        '''
        将list的4个array打散,变成融合图的坐标
        :param bboxes:list ,长度4,每个元素为array数组,[num_target,5]
        :param cutx: 中间点位的x
        :param cuty: 中间点位的y
        :return:
        '''
        merge_bbox = []
        for i in range(len(bboxes)):
            for box in bboxes[i]:
                # 对每个target处理
                tmp_box = []
                x1, y1, x2, y2 = box[0], box[1], box[2], box[3]

                # 新增,过滤宽高小于最低阈值的框框
                # 小于这个阈值连一个像素都不够,肯定无法预测
                if (y2 - y1<=threshold) or (x2-x1<=threshold):
                    continue
                if i == 0:
                    if y1 > cuty or x1 > cutx:
                        continue
                    if y2 >= cuty and y1 <= cuty:
                        y2 = cuty
                    if x2 >= cutx and x1 <= cutx:
                        x2 = cutx

                if i == 1:
                    if y2 < cuty or x1 > cutx:
                        continue
                    if y2 >= cuty and y1 <= cuty:
                        y1 = cuty
                    if x2 >= cutx and x1 <= cutx:
                        x2 = cutx

                if i == 2:
                    if y2 < cuty or x2 < cutx:
                        continue
                    if y2 >= cuty and y1 <= cuty:
                        y1 = cuty
                    if x2 >= cutx and x1 <= cutx:
                        x1 = cutx

                if i == 3:
                    if y1 > cuty or x2 < cutx:
                        continue
                    if y2 >= cuty and y1 <= cuty:
                        y2 = cuty
                    if x2 >= cutx and x1 <= cutx:
                        x1 = cutx
                tmp_box.append(x1)
                tmp_box.append(y1)
                tmp_box.append(x2)
                tmp_box.append(y2)
                tmp_box.append(box[-1])
                merge_bbox.append(tmp_box)
        return merge_bbox

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
        nws     = [ int(w * self.rand(0.4, 1)), int(w * self.rand(0.4, 1)), int(w * self.rand(0.4, 1)), int(w * self.rand(0.4, 1))]
        nhs     = [ int(h * self.rand(0.4, 1)), int(h * self.rand(0.4, 1)), int(h * self.rand(0.4, 1)), int(h * self.rand(0.4, 1))]

        # 图片放置的左上角xy坐标 顺序为
        # 1 4
        # 2 3
        # 注意palce_x和place_y都是像素单位的坐标
        place_x = [int(w*min_offset_x) - nws[0], int(w*min_offset_x) - nws[1], int(w*min_offset_x), int(w*min_offset_x)]
        place_y = [int(h*min_offset_y) - nhs[0], int(h*min_offset_y), int(h*min_offset_y), int(h*min_offset_y) - nhs[3]]

        image_datas = [] 
        box_datas   = []
        index       = 0
        for line in annotation_line:
            # 每一行进行分割
            line_content = line.split()
            # 打开图片
            image = Image.open(line_content[0])
            image = cvtColor(image)
            
            # 图片的大小
            iw, ih = image.size
            # 保存框的位置 box似乎是个5维,最后一维为类别
            box = np.array([np.array(list(map(int,box.split(',')))) for box in line_content[1:]])
            
            # 是否翻转图片
            flip = self.rand()<.5
            if flip and len(box)>0:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
                box[:, [0,2]] = iw - box[:, [2,0]]

            nw = nws[index] 
            nh = nhs[index] 
            image = image.resize((nw,nh), Image.BICUBIC)

            # 将图片进行放置，分别对应四张分割图片的位置
            dx = place_x[index]
            dy = place_y[index]
            if self.rand() < 0.5:
                color = (0, 0, 0)
            elif self.rand() < 1:
                color = (128, 128, 128)
            else:
                color = (128, 128, 128)
            new_image = Image.new('RGB', (w,h), color)
            new_image.paste(image, (dx, dy))
            image_data = np.array(new_image)

            index = index + 1
            box_data = []
            # 对box进行重新处理
            if len(box)>0:
                np.random.shuffle(box)
                box[:, [0,2]] = box[:, [0,2]]*nw/iw + dx
                box[:, [1,3]] = box[:, [1,3]]*nh/ih + dy
                box[:, 0:2][box[:, 0:2]<0] = 0
                box[:, 2][box[:, 2]>w] = w
                box[:, 3][box[:, 3]>h] = h
                box_w = box[:, 2] - box[:, 0]
                box_h = box[:, 3] - box[:, 1]
                # 删除框框宽高<1的,以及框框部分超过整张[416,416]图片的
                box = box[np.logical_and(box_w>1, box_h>1)]
                box_data = np.zeros((len(box),5))
                box_data[:len(box)] = box
            # 每个image_data或box_data中存一个只有1/4的图片和位置信息?
            image_datas.append(image_data)
            box_datas.append(box_data)

        # 将图片分割，放在一起
        cutx = int(w * min_offset_x)
        cuty = int(h * min_offset_y)

        # 这样申请出来的new_image竟然是float64数据,奇葩
        new_image = np.zeros([h, w, 3])
        # 注意,因为从Image变为array的过程中,wh发生变换
        # 如Image中是(w,h)
        # array中则是(h,w,channel)
        # 所以下面写的奇怪,是先取y轴,再取x轴
        new_image[:cuty, :cutx, :] = image_datas[0][:cuty, :cutx, :]
        new_image[cuty:, :cutx, :] = image_datas[1][cuty:, :cutx, :]
        new_image[cuty:, cutx:, :] = image_datas[2][cuty:, cutx:, :]
        new_image[:cuty, cutx:, :] = image_datas[3][:cuty, cutx:, :]

        # 进行色域变换
        hue = self.rand(-hue, hue)
        sat = self.rand(1, sat) if self.rand()<.5 else 1/self.rand(1, sat)
        val = self.rand(1, val) if self.rand()<.5 else 1/self.rand(1, val)
        x = cv2.cvtColor(np.array(new_image/255,np.float32), cv2.COLOR_RGB2HSV)
        x[..., 0] += hue*360
        x[..., 0][x[..., 0]>1] -= 1
        x[..., 0][x[..., 0]<0] += 1
        x[..., 1] *= sat
        x[..., 2] *= val
        x[x[:, :, 0]>360, 0] = 360
        x[:, :, 1:][x[:, :, 1:]>1] = 1
        x[x<0] = 0
        new_image = cv2.cvtColor(x, cv2.COLOR_HSV2RGB)*255

        # 对框进行进一步的处理
        # 再把图片重叠在其他区域的框框删除掉,以及将4个图片的框框融合成1个图片的
        # 32是因为yolo默认的416的输入会缩放到13,26,52特征图是缩放到,会缩放到
        # 分别对应原图上的32,16,8个像素
        # 最小的为8,小于8个像素的目标可能经过切割,因此效果不好
        # 标定没有意义
        minest_size=8
        new_boxes = self.merge_bboxes(box_datas, cutx, cuty,minest_size)

        return new_image, new_boxes

# DataLoader中collate_fn使用
def yolo_dataset_collate(batch):
    images = []
    bboxes = []
    for img, box in batch:
        images.append(img)
        bboxes.append(box)
    images = np.array(images)
    return images, bboxes