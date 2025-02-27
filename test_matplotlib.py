import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
import time

# fig, ax = plt.subplots()
# x = np.linspace(0, 2*np.pi, 100)
# line, = ax.plot(x, np.sin(x))
#
# def update(frame):
#     line.set_ydata(np.sin(x + frame / 10))
#     return line,
#
# ani = FuncAnimation(fig, update, frames=100, interval=50)
# plt.show()



# # 初始化数据
# x = []
# y = []
#
# # 创建画布和子图
# fig, ax = plt.subplots(figsize=(8, 6))
#
# for i in range(10):
#     # 添加新数据点
#     x.append(i)
#     y.append(np.sin(i))
#
#     # 清空当前子图并重绘
#     ax.clear()
#     ax.plot(x, y, 'b-')
#     ax.set_title(f'Step {i + 1}')
#     ax.set_xlim(0, 10)
#     ax.set_ylim(-1.5, 1.5)
#
#     # 更新显示
#     plt.pause(0.5)  # 暂停0.5秒以模拟实时效果
#
# plt.show()


# # 初始化数据
# x = []
# y = []
#
# # 创建画布和初始空折线
# fig, ax = plt.subplots(figsize=(8, 6))
# line, = ax.plot([], [], 'b-')  # 返回一个 Line2D 对象
# ax.set_xlim(0, 10)
# ax.set_ylim(-1.5, 1.5)
#
# # 循环更新
# for i in range(10):
#     x.append(i)
#     y.append(np.sin(i))
#
#     # 更新折线数据
#     line.set_data(x, y)
#     ax.set_title(f'Step {i + 1}')
#
#     # 刷新画布
#     fig.canvas.draw()
#     fig.canvas.flush_events()  # 确保更新生效
#     plt.pause(0.5)
#
# plt.show()



# # 初始化数据
# x = []
# y = []
#
# # 创建画布和初始空折线
# fig, ax = plt.subplots(figsize=(8, 6))
# line, = ax.plot([], [], 'b-')
# ax.set_xlim(0, 10)
# ax.set_ylim(-1.5, 1.5)
#
# # 更新函数
# def update(frame):
#     x.append(frame)
#     y.append(np.sin(frame))
#     line.set_data(x, y)
#     ax.set_title(f'Step {frame+1}')
#     return line,
#
# # 创建动画
# ani = FuncAnimation(fig, update, frames=range(10), interval=500, blit=True)
#
# plt.show()


# 初始化数据
data1 = {'x': [], 'y': []}
data2 = {'x': [], 'y': []}

# 创建两个画布
fig1, (ax11, ax12) = plt.subplots(2, 1, figsize=(8, 6))
fig2, ax2 = plt.subplots(figsize=(8, 6))

# 初始化折线
line11, = ax11.plot([], [], 'b-')
line12, = ax12.plot([], [], 'r-')
line2, = ax2.plot([], [], 'g-')

# 设置坐标轴范围
for ax in [ax11, ax12, ax2]:
    ax.set_xlim(0, 10)
    ax.set_ylim(-1.5, 1.5)


# 更新函数
def update(frame):
    data1['x'].append(frame)
    data1['y'].append(np.sin(frame))
    data2['x'].append(frame)
    data2['y'].append(np.cos(frame))

    line11.set_data(data1['x'], data1['y'])
    line12.set_data(data1['x'], data2['y'])
    line2.set_data(data2['x'], data2['y'])

    ax11.set_title(f'Figure 1 - Step {frame + 1}')
    return line11, line12, line2


# 动画仅绑定一个 Figure，但更新所有
ani = FuncAnimation(fig1, update, frames=range(10), interval=500, blit=True)

plt.show()

