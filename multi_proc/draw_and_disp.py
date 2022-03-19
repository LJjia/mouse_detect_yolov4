#!/usr/bin/env python
#-*- coding:utf-8 _*-
__author__ = 'LJjia'
# *******************************************************************
#     Filename @  draw_and_disp.py
#       Author @  Jia Liangjun
#  Create date @  2022/03/18 18:50
#        Email @  LJjiahf@163.com
#  Description @  绘制坐标框并且显示
# ********************************************************************

from multiprocessing import Process,Queue
import time
import cv2
import os
from utils.utils import (cvtColor, get_anchors, get_classes, preprocess_input,
                         resize_image)

class MainProcess(Process): #继承Process类
    def __init__(self,name,msg_que):
        super(MainProcess,self).__init__()
        self.name = name
        self.queue=msg_que
    def run(self):

        print('初始化queue',self.name)
        cnt=100
        while(not self.queue.full()):
            cnt+=1
            self.queue.put('que'+str(cnt))
        print('=====main queue fill ok! ====')

class SubProcess(Process): #继承Process类
    def __init__(self,name,msg_que):
        super(SubProcess,self).__init__()
        self.name = name
        self.queue=msg_que
    def run(self):
        while(not self.queue.empty()):
            data=self.queue.get()
            print('queue',self.name,'pop data ',data)
        print('sub queue pop end')


class VideoDecodeProc(Process):
    '''
    视频解码对象
    '''
    _defaults = {
        # 输入到网络的图片大小
        "input_shape": [608, 608],
        # 是否填黑边,True表示填黑边,False表示直接失真的resize
        "letterbox_image":True,
    }
    s_prco_cnt=0
    def __init__(self,video_path,dst_que,name='VideoDecodeProc',wait_time=3):
        super().__init__()
        self.name=name+'_'+str(VideoDecodeProc.s_prco_cnt)
        VideoDecodeProc.s_prco_cnt+=1

        if (video_path!=0) and (not os.path.exists(video_path)):
            raise FileExistsError("%s not found" % video_path)
        self.capture = cv2.VideoCapture(video_path)
        self.proc_frames_cnt=0
        self.que=dst_que
        self.wait_time=wait_time

    def run(self):
        print('run prco ',self.name)
        while True:
            ref, image = self.capture.read()

            if not ref:
                if self.proc_frames_cnt==0:
                    raise ValueError("未能正确读取摄像头（视频），请注意是否正确安装摄像头（是否正确填写视频路径）。")
                else:
                    # 证明处理完了
                    print('video prco finish ', self.proc_frames_cnt)
                    break
            self.proc_frames_cnt += 1
            image = cvtColor(image)
            image_data = resize_image(image, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)
            self.que.put(image_data,timeout=self.wait_time)


class DispImgProc(Process):
    '''
    图片显示对象
    '''
    _defaults = {
        # 输入到网络的图片大小
        "input_shape": [608, 608],
        # 是否填黑边,True表示填黑边,False表示直接失真的resize
        "letterbox_image":True,
    }
    s_prco_cnt=0
    def __init__(self, src_que, name='DispImgProc', wait_time=5):
        super().__init__()
        self.name = name + '_' + str(VideoDecodeProc.s_prco_cnt)
        VideoDecodeProc.s_prco_cnt += 1

        # 每30帧更新一下fsp
        self.fps_period=30
        # 每个周期的总时长 单位 秒
        self.frames_period_interval = self.fps_period
        self.disp_frames_total=0
        self.que = src_que
        self.wait_time=wait_time
    def run(self):
        img,result=self.que.get(timeout=self.wait_time)




if __name__ == '__main__':
    # process_list = []
    # main_que=Queue(80)
    # p = MainProcess('Python' + str(0),main_que)  # 实例化进程对象
    # p.start()
    # time.sleep(1)
    #
    # for i in range(1,5):  #开启5个子进程执行fun1函数
    #     p = SubProcess('python'+str(i),main_que) #实例化进程对象
    #     p.start()
    #     process_list.append(p)
    #
    # for i in process_list:
    #     p.join()
    pass

    print('结束测试')


