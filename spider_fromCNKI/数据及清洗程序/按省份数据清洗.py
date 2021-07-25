# !/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time : 2020/5/28 20:36
# @Author : Mat
# @Email : mat_wu@163.com
# @File : data_process.py
# @Software: PyCharm

import numpy as np
import pandas as pd
import os
import json
import jmespath
import csv

f = open("1622205159175.json",'r',encoding='utf-8')
provice = json.load(f)                               # 读取省市县文件列表
# print(len(provice))
#

def proVince(x):                                                  # 定义省份判断函数
    for j in range(len(provice)):
        result3 = provice[j]['name']                              # 遍历每一个省的省份名字
        result4 = jmespath.search('city[*].name', provice[j])     # 遍历每一个省的地级市
        result5 = jmespath.search('city[*].area', provice[j])     # 遍历每一个地级市下的县，区等地名
        result6 = []
        result6.append(result3)
        for kk in result4:
            result6.append(kk)
        for h in result5:
            for k in h:
                result6.append(k)
        # print(result6)
        for ii in result6:
            # if ii == 1:
            #     print(result6[ii])
            if ii[0:2] in x:                   # x为爬取的数据的from来源，即省份等地名信息，判断该省份属于上述文件列表中的哪个省份
                return result3                 # 当判断出属于某个省份时，返回数据。



with open('zhiwang6.json', 'r',encoding='utf-8') as fp:  # 读取爬取数据文件，找到爬取数据
    fg = fp.readlines()

print(eval(fg[0])['title'])
cvp = []
for i in range(len(fg)):
    froM = eval(fg[i])['from']                          # 找到from中省市信息产业文件来源
    print(froM)
    result3 = proVince(froM)             # 调用省份判别函数，得到返回结果
    print(result3)
    if result3 != None:
        with open(result3+".csv", "a", newline="",encoding='utf_8_sig') as apdFile:
            w = csv.writer(apdFile)
            w.writerow([eval(fg[i])['title'],eval(fg[i])['from'],eval(fg[i])['date'],eval(fg[i])['db'],eval(fg[i])['level']])   # 按省份保存结果
    else:
        with open('shengyu.json', 'a+', encoding='utf-8') as fp:    # 保存不能识别的地名信息文件
            fp.write(fg[i] + '\n')



