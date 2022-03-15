#!/usr/bin/env python
#-*- coding:utf-8 _*-
__author__ = 'LJjia'
# *******************************************************************
#     Filename @  track.py
#       Author @  Jia Liangjun
#  Create date @  2022/03/15 12:27
#        Email @  LJjiahf@163.com
#  Description @  
# ********************************************************************

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

fig = plt.figure()  # 创建figure对象
ax = plt.axes()  # 极坐标系
ax.set_axis_off()  # 取消坐标轴的显示
ln, = ax.plot([], [])

# 图像初始化
def init():
    ax.set_xlim(0, 2*np.pi)       # 设定x值范围
    ax.set_ylim(-1, 1)            # 设定y值范围
    xdata = [1,2,3,4]
    ydata = [0, 0, 0, 0]
    ln.set_data(xdata, ydata)
    return ln,

# 图像更新
def update(frame):
    xdata = [1*frame, 2*frame, 3*frame, 4*frame]
    ydata = [0,0,0,0]
    ln.set_data(xdata, ydata)
    return ln,

ani = FuncAnimation(fig, update, frames=np.linspace(0, 2*np.pi, 128),
                    init_func=init, blit=True, interval=1)
plt.show()
# saves the animation in our desktop
# anim.save('growingCoil.mp4', writer='ffmpeg', fps=30)