#!/usr/bin/env python
#-*- coding:utf-8 _*-
__author__ = 'LJjia'
# *******************************************************************
#     Filename @  draw_result.py
#       Author @  Jia Liangjun
#  Create date @  2022/03/15 11:15
#        Email @  LJjiahf@163.com
#  Description @  绘制轨迹图,主要显示卡尔满滤波前后效果,因此只有绘制单个轨迹,不判断obj的ID
# ********************************************************************

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation



def str2int(s):
    return int(float(s))

def get_result(src_path):
    with open(src_path,'r') as f:
        lines=f.readlines()
    # 创建0行2列矩阵
    ret=np.empty((0,2),dtype=np.int32)
    for line in lines:
        line=line.strip().split(',')
        pos=list(map(str2int,line))[:4]
        # print(pos)
        center=np.array([int((pos[2]+pos[0])/2),int((pos[3]+pos[1])/2)])
        # print(center[None,:])
        ret=np.append(ret,center[None,:],axis=0)

    # print(ret,ret.shape)
    return ret

class PlotFig(object):
    def __init__(self,x=0,y=0,title="animation"):
        self.x_range=(0,100)
        self.y_range=(50,700)
        self.xx=[]
        self.yy=[]

        self.fig = plt.figure(tight_layout=True)

        # 设置坐标范围
        axis = plt.axes(xlim=(self.x_range[0], self.x_range[1]),
                        ylim=(self.y_range[0], self.y_range[1]))
        axis.xaxis.set_ticks_position('top')  # 将x轴的位置设置在顶部
        axis.invert_yaxis()  # y轴反向
        plt.title(title)
        plt.grid()
        self.point_ani, = axis.plot([], [], "r-", lw=1)  # 必须有,，表示得到元组
        self.text_pt = plt.text(0, 0, '', fontsize=8)
        # 设置文本位置
        self.text_pt.set_position((self.x_range[1] - 120, self.y_range[0] + 20))
        self.cnt=0
    def ani_init(self):
        self.point_ani.set_data([], [])
        return self.point_ani,
    def update(self,data):
        '''更新数据点
        .set_data()的意思是将这里的(x[num], y[num])代替上面的(x[0], y[0])
        也可以.set_ydata,需要将上面的x[0]改成x,这里的x[num]去掉
        '''
        self.cnt+=1
        # print(self.cnt)
        self.xx.append(int(data[0]))
        self.yy.append(int(data[1]))
        self.point_ani.set_data(self.xx, self.yy)
        # 更新文本内容
        self.text_pt.set_text("x=%3d, y=%3d" % (data[0], data[1]))
        return self.point_ani,self.text_pt

    def feed(self,data,save_file=None):
        print('input shape', data.shape)
        # 开始制作动画
        ani = animation.FuncAnimation(fig=self.fig, func=self.update,init_func=self.ani_init,
                                      frames=data,interval=2, blit=False,repeat=False)

        if(save_file!=None):
            # 保存gif
            # pw_writer = animation.PillowWriter(fps=60)
            # ani.save(save_file, writer=pw_writer)
            # 保存视频
            ani.save(save_file, writer='ffmpeg',fps=25)
        else:
            plt.show()








if __name__ == '__main__':
    src_res = 'kalman.txt'
    data=get_result(src_res)
    ani=PlotFig(title='kalman')
    ani.feed(data,'kalman.mp4')
