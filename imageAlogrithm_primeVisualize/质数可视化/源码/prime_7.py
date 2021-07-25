# !/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time : 2021/5/22 17:30
# @Author : Mat
# @Email : mat_wu@163.com
# @File : prime_7.py
# @Software: PyCharm

import matplotlib.pyplot as plt
from math import sqrt
import math
import numpy as np

def is_prime(n):
    x = [p for p in range(2, n+1) if 0 not in [p % d for d in range(2, int(sqrt(p)) + 1)]]
    print(x)
    return x

primesList = is_prime(1000) # 判断1000以内的素数
A = np.zeros((1000,2)) # 生成（1000X2）的零矩阵
z = -1 #复数z决定前进方向，初始值决定起始方向

w = math.exp(1/4*math.pi)  # 设置w复数 w决定质数矩阵变换
for i in range(2,1000):
    if i in primesList:
        z = z * w
    A[i, 0] = A[i-1, 0] + z.real
    A[i, 1] = A[i-1, 1] + z.imag
print(A)
for i in range(1000):
    plt.scatter(A[i,0],A[i,1])   # 绘制图形
plt.show()
