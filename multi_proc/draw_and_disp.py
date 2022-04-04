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
import colorsys
import os
import numpy as np
from track.sort import Sort
from PIL import ImageDraw, ImageFont,Image
from utils.utils import (cvtColor, get_anchors, get_classes, preprocess_input,
                         resize_image)

'''
以下两个类用作测试
'''
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
    def __init__(self,video_path,dst_que,capture,name='VideoDecodeProc',wait_time=3):
        super().__init__()
        self.name=name+'_'+str(VideoDecodeProc.s_prco_cnt)
        VideoDecodeProc.s_prco_cnt+=1
        self.__dict__.update(self._defaults)

        if (video_path!=0) and (not os.path.exists(video_path)):
            raise FileExistsError("%s not found" % video_path)
        self.capture = capture
        self.proc_frames_cnt=0
        self.que=dst_que
        self.wait_time=wait_time

    def run(self):
        print('run proc ',self.name)
        while True:
            ref, frame = self.capture.read()

            if not ref:
                if self.proc_frames_cnt==0:
                    raise ValueError("未能正确读取摄像头（视频），请注意是否正确安装摄像头（是否正确填写视频路径）。")
                else:
                    # 证明处理完了
                    print('video prco finish ', self.proc_frames_cnt)
                    break
            self.proc_frames_cnt += 1
            # bgr2rgb
            frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            # 转变成Image
            frame = Image.fromarray(np.uint8(frame))
            # image格式的图片
            self.que.put(frame, timeout=self.wait_time)



class DispImgProc(Process):
    '''
    图片显示对象
    '''
    _defaults = {
        # 输入到网络的图片大小
        "input_shape": [608, 608],
        # 是否填黑边,True表示填黑边,False表示直接失真的resize
        "letterbox_image":True,
        "classes_path": 'model_data/voc_classes.txt',
    }
    s_prco_cnt=0
    def __init__(self, src_que, name='DispImgProc', wait_time=5,prt_msg=False,dump2file=None):
        super().__init__()
        self.name = name + '_' + str(VideoDecodeProc.s_prco_cnt)
        VideoDecodeProc.s_prco_cnt += 1
        self.__dict__.update(self._defaults)

        # 每30帧更新一下fsp
        self.fps_period=30
        # 每个周期的总时长 单位 秒
        self.frames_period_interval = self.fps_period
        self.disp_frames_total=0
        self.que = src_que
        self.wait_time=wait_time
        self.kalman=Sort()
        self.prt_msg=prt_msg
        if dump2file:
            # 重开一个文件记录
            with open(dump2file,'w') as f:
                pass
        self.dump2file=dump2file
        self.class_names, self.num_classes = get_classes(self.classes_path)
        # ---------------------------------------------------#
        #   画框设置不同的颜色
        # ---------------------------------------------------#
        hsv_tuples = [(x / self.num_classes, 1., 1.) for x in range(self.num_classes)]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))

    def run(self):
        print('run proc ', self.name)
        fps = 0
        while True:

            if self.disp_frames_total==0:
                start_time=time.time()
            # Image格式的img,和np格式的result,如果没有数据,result为None
            frame,result,idx=self.que.get(timeout=self.wait_time)
            # frame=Image.fromarray(frame)
            # print(result)

            # time.sleep(0.01)
            if not (result is None):
                frame = self.draw_on_img(frame, result)
            else:
                pass
            frame=np.array(frame)
            # RGBtoBGR满足opencv显示格式
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)


            frame = cv2.putText(frame, "fps= %.2f" % (fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow("video", frame)
            c = cv2.waitKey(1) & 0xff
            if c == 27:
                self.capture.release()
                break
            self.disp_frames_total += 1
            time.sleep(0.01)
            if self.disp_frames_total==self.fps_period:
                fps=(1./(time.time()-start_time))*self.fps_period
                self.disp_frames_total=0
                # 视频不打印fps,直接在图上显示
                # print('dips fps %2f', fps)


        print("Video Detection Done!")
        cv2.destroyAllWindows()

    def draw_on_img(self,image,result):
        '''

        :param img: RGB numpy 格式
        :param result:numpy array obj坐标
        :return: RGB numpy格式
        '''
        # image=Image.fromarray(image)
        top_label = np.array(result[:, 6], dtype='int32')
        top_conf = result[:, 4] * result[:, 5]
        top_boxes = result[:, :4]
        # ---------------------------------------------------------#
        #   设置字体与边框厚度
        # ---------------------------------------------------------#
        font = ImageFont.truetype(font='model_data/simhei.ttf',
                                  size=np.floor(4e-2 * image.size[1] + 0.5).astype('int32'))
        thickness = int(max((image.size[0] + image.size[1]) // np.mean(image.size), 1))

        # 本来top_boxes和top_conf就是numpy类型
        dets = np.concatenate((top_boxes, top_conf[:, None]), axis=1)
        # print('input',dets)
        res = self.kalman.update(dets)
        # print('out', res)
        top_boxes = res[:, :4]
        obj_ids = res[:, 4]

        for i, _ in list(enumerate(top_boxes)):
            c=top_label[i]
            predicted_class = self.class_names[int(c)]
            box             = top_boxes[i]
            score           = top_conf[i]
            obj_id=int(obj_ids[i])

            top, left, bottom, right = box

            top     = max(0, np.floor(top).astype('int32'))
            left    = max(0, np.floor(left).astype('int32'))
            bottom  = min(image.size[1], np.floor(bottom).astype('int32'))
            right   = min(image.size[0], np.floor(right).astype('int32'))

            label = '{}_ID{}_{:.2f}'.format(predicted_class, obj_id,score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)
            label = label.encode('utf-8')
            # 是否打印目标
            if self.prt_msg:
                print(label, top, left, bottom, right)
            if self.dump2file:
                with open(self.dump2file, 'a') as file:
                    corrd_str=str(int(top + bottom)) + ',' + str(int(left + right))+','+str(score)+'\n'
                    file.write(corrd_str)

            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            for i in range(thickness):
                draw.rectangle([left + i, top + i, right - i, bottom - i], outline=self.colors[c])
            draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=self.colors[c])
            draw.text(text_origin, str(label,'UTF-8'), fill=(0, 0, 0), font=font)
        return np.array(image)








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

    print('结束测试')


