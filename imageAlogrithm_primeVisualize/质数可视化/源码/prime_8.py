# !/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time : 2021/5/22 19:43
# @Author : Mat
# @Email : mat_wu@163.com
# @File : prime_8.py
# @Software: PyCharm

import matplotlib.pyplot as plt
from math import sqrt
import math
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # 空间三维画图


def is_prime(n):    # 定义判断是否为素数函数
    x = [p for p in range(2, n+1) if 0 not in [p % d for d in range(2, int(sqrt(p)) + 1)]]
    return x
x = []
for i in range(1,20):  # 设置20行自然数
    x1 = 6*i            # 每行自然数个数
    x.append(x1)
xa1 = []               # 质数矩阵行下标
ya1 = []# 质数矩阵列下标
ca1 = []# 质数原来的行下标
xa2 = []# 合数矩阵行下标
ya2 = []# 合数矩阵列下标
ca2 = []# 合数原来的行下标
sumx = sum(x)  # 求解总的自然数个数
print(sumx)
cv = is_prime(sumx) # 生成素数列表
n = []
mm = []
for i in range (sumx):
    n.append(i)        # 保存自然数列表
for i in range (sumx):
    for j in range(len(x)):
        sumc = np.array(n[(j+1)*3*j:(x[j]+((j+1)*3*j))]).reshape(6,j+1)  # 对每一行自然数切片并重新reshape
        mm.append(sumc)
        for ii in range(sumc.shape[0]):
            for jj in range(sumc.shape[1]):

                if sumc[ii,jj] in cv:
                    xa1.append(jj)
                    ca1.append(ii)
                    ya1.append(j)
                    # plt.scatter(jj,ii,j)
                else:
                    xa2.append(jj)
                    ca2.append(ii)
                    ya2.append(j)
                    # plt.scatter(jj,ii,j)
# plt.show()
fig = plt.figure()
plt.scatter(xa1, ca1)  # 绘制质数矩阵行坐标，所属原来行坐标
plt.scatter(xa2, ca2) # 绘制合数矩阵行坐标，所属原来行坐标
plt.scatter(xa1, ya1)  # 绘制质数矩阵行坐标，质数矩阵列坐标
plt.scatter(xa2, ya2) # 绘制合数矩阵行坐标，合数矩阵列坐标
plt.scatter(ca1, ya1)  # 所属原来行坐标，质数矩阵列坐标
plt.scatter(ca2, ya2) # 绘制原来行坐标，合数矩阵列坐标

plt.show()