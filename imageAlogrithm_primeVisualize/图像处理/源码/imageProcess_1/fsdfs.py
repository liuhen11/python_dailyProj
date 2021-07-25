import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import math
from PIL import Image
from tqdm import tqdm
import cv2
import kok      # 这个是我自己定义的面积计算模块，在另一个名为kok.py的文件里
import os

i = 100
cov = 0
N=0
n=0
R = 0.5
r = 0.06
# cv2.imread 读取图像函数
img2=cv2.imread("./roitiqu.png") #  roitiqu.png图像 目的主要是为了提取出指定大圆区域内的图像信息，以方便计算大圆内小圆所占阴影总面积
rows,cols,channels=img2.shape    # 获得图像的大小，rgb通道数
img2gray = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)  # 将img2 转为灰度图
ret,mask = cv2.threshold(img2gray,175,255,cv2.THRESH_BINARY)  # 二值化，提取出红色大圆掩码区域 mask即为掩码信息
mask_inv = cv2.bitwise_not(mask)                             # 掩码非运算


list_kn = []                                                # 创建空列表，保存500次所有的小圆和符合条件的小圆的个数

for p in tqdm(range(0, 500)):                               # 500 次循环
    mean = np.array([0, 0])                                 # 均值
    conv = np.array([[0.259, 0.0], [0, 0.259]])            # 方差？这个您应该清楚
    s = 0                                                  # 初始面积为0
    redl = 0                                               # 红色初始面积为0
    blackl = 0                                             # 黑色初始面积为0
    fig = plt.figure(figsize=(4, 3), dpi=200)              # 设置图像尺寸大小 4*200=800，3*200=600
    ax = fig.add_subplot()                                 # 设置子图
    c = Circle(xy=(0, 0), radius=0.5, facecolor=[1, 0, 0]) # 绘制大圆
    ax.add_patch(c)                                        # 将大圆添加进子图
    plt.axis("equal")                                      # 设置横坐标和纵坐标等宽显示
    plt.axis("off")                                        # 关闭坐标轴，如果想打开，直接删除或off改为on
    while s <= 0.98:                                       # 判断 面积s是否小于0.98
        x, y = np.random.multivariate_normal(mean=mean, cov=conv, size=1).T  # 生成小圆圆心坐标
        z = [x[0], y[0]]                                                     # 将坐标变成列表
        c1 = Circle(xy=z, radius=0.06, facecolor=[0, 0, 0])                  # 画小圆
        ax.add_patch(c1)                                                     # 将小圆添加进子图
        N = N + 1                                                            # 总圆个数
        plt.savefig('./test.png')                                            # 保存所有小圆图片
        d = math.sqrt(z[0] * z[0] + z[1] * z[1])                             # 计算圆心距
        if d <= 0.5:
            n = n + 1                                                        # 符合条件的小圆个数加1
            img1 = cv2.imread('./test.png')                                  # 读取之前图片
            roi = img1[0:rows, 0:cols]                                       # 这一步的目的主要是为了提取出img1中但大小是img1大小的图像内容
            img1_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)               # 这一步主要是提取出 test大圆内的所有图像信息
            img2_fg = cv2.bitwise_and(img2, img2, mask=mask)                 # 这一步提取出img2中的背景色，即所有的白色区域
            dst = cv2.add(img1_bg, img2_fg)                                  # 将白色背景区域和test大圆内区域合成，就可得到新的tesr信息
            img1[0:rows, 0:cols] = dst                                       # 将tesr信息覆盖到img1中 ，以上这几步不明白可以一步一步执行，参考Untitled1.ipynb
            imfg = Image.fromarray(img1[..., ::-1])                         # 因为cv2读取的图片是gbr图像，采用-1转换成rgb图像
            imfg.save('./tesr.png')                                         # 保存tesr
            s, cvb = kok.ploro()                                             # 调用kok.py文件中的ploro函数
            print(f'小圆个数为{n}')
    print(f'第{p}次循环执行完成')
    list_kn.append((N, n))                                                         # 添加进上述建立的空列表
    with open('总的小圆个数和符合条件小圆个数.txt','a') as f:                                   # 将保存的结果写进txt文件
        f.writelines(f'第{p}次循环'+ '  '+ str(N)+ '  ' + str(n))                  # 每一次大循环的结果
        f.writelines('\n')                                                        # 进行换行操作
    N = 0
    n = 0
    plt.imshow(cvb)
print(list_kn)