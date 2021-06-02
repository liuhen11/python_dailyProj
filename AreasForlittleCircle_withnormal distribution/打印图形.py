# !/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time : 2021/4/2 20:59
# @Author : Mat
# @Email : mat_wu@163.com
# @File : 打印图形.py
# @Software: PyCharm

n=int(9/2)+1#计算行数
m=1
for i in range(n):
  print((' '*((9-m)//2))+('*'*m)+(' '*((9-m)//2)))#((num-m)//2)是‘*’前后的空格数
  m+=2

#a=int(input('输入奇数'))
for i in range(1,9+1,2):
    t = (9-i)//2
    print(' '*t + '*'*i)
for y in range(7,0,-2):
    x = (9-y)//2
    print(' '*x+y*'*')
