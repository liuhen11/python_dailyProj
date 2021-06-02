# !/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time : 2021/5/9 22:19
# @Author : Mat
# @Email : mat_wu@163.com
# @File : tiananmen.py
# @Software: PyCharm
import turtle as t
import random as r
#pen = t.Pen()


#位移函数
def Skip(t,x,y):
    t.penup()
    t.goto(x,y)
    t.pendown()

def pink():
    color = (1, r.random(), 1)
    return color

#画笔基础设置
t.screensize(1200,800)
t.pensize(5)
t.hideturtle()
t.speed(20)
t.pencolor("red")
startx, starty = -150, 50
Skip(t,-350,200)
t.pencolor("black")
t.write("百年建党，我爱重财", font=("楷体", 50, "bold"))

t.pencolor("red")
#画笔移动
Skip(t,-200,100)

#画房盖
t.fillcolor(1,1,0)
t.begin_fill()
t.circle(40,90)
t.right(90)
t.forward(200)
t.right(90)
t.circle(40,90)
t.right(180)
t.forward(280)
t.end_fill()

#顶层
t.fillcolor(1,1,0)
t.begin_fill()
t.left(135)
t.forward(20)
t.left(45)
t.forward(252)
t.left(45)
t.forward(20)
t.end_fill()

t.fillcolor('#FFA500')
t.begin_fill()
Skip(t,-184,82)
t.right(135)
t.forward(20)
t.left(90)
t.forward(249)
t.left(90)
t.forward(20)
t.left(90)
t.forward(249)
t.end_fill()

#第二层屋檐
t.fillcolor(1,1,0)
t.begin_fill()
Skip(t,-184,62)
t.left(20)
t.forward(50)
t.circle(-40,50)
t.left(150)
t.circle(30,60)
t.forward(354)
t.circle(30,60)
t.left(150)
t.circle(-40,50)
t.forward(50)
t.left(20)
t.forward(249)
t.end_fill()

#第二层
t.fillcolor('#FFA500')
t.begin_fill()
Skip(t,-214,33)
t.left(90)
t.forward(30)
t.left(90)
t.forward(309)
t.left(90)
t.forward(30)
t.left(90)
t.forward(309)
t.end_fill()

#第二层柱子
t.left(90)
Skip(t,-183,33)
t.forward(30)
Skip(t,-152,33)
t.forward(30)
Skip(t,-121,33)
t.forward(30)
Skip(t,-90,33)
t.forward(30)
Skip(t,-59,33)
t.forward(30)
Skip(t,-28,33)
t.forward(30)
Skip(t,3,33)
t.forward(30)
Skip(t,34,33)
t.forward(30)
Skip(t,65,33)
t.forward(30)
t.left(180)

#外墙
Skip(t,-214,3)
t.left(90)
t.forward(250)
t.left(90)
t.forward(100)
t.left(90)
t.forward(809)
t.left(90)
t.forward(100)
t.left(90)
t.forward(250)
t.fillcolor('#b3afaf')
t.begin_fill()
Skip(t,-464,-15)
t.left(180)
t.forward(383)
t.left(90)
t.forward(11)
t.left(270)
t.forward(44)
t.left(270)
t.forward(11)
t.left(90)
t.forward(383)
t.left(90)
t.forward(18)
t.left(90)
t.forward(809)
t.left(90)
t.forward(18)
t.end_fill()
#
# Skip(t,-37,-15)
# t.forward(383)
# t.end_fill()

#正门和侧门
t.fillcolor('#980000')
t.begin_fill()
Skip(t,-79,-97)
t.left(180)
t.forward(15)
t.circle(-20,180)
t.forward(15)
t.left(270)
t.forward(40)
t.end_fill()

t.fillcolor('#980000')
t.begin_fill()
Skip(t,-189,-97)
t.left(270)
t.forward(10)
t.circle(-15,180)
t.forward(10)
t.left(270)
t.forward(30)
t.end_fill()

t.fillcolor('#980000')
t.begin_fill()
Skip(t,31,-97)
t.left(270)
t.forward(10)
t.circle(-15,180)
t.forward(10)
t.left(270)
t.forward(30)
t.end_fill()

t.fillcolor('#980000')
t.begin_fill()
Skip(t,-269,-97)
t.left(270)
t.forward(10)
t.circle(-15,180)
t.forward(10)
t.left(270)
t.forward(30)
t.end_fill()

t.fillcolor('#980000')
t.begin_fill()
Skip(t,111,-97)
t.left(270)
t.forward(10)
t.circle(-15,180)
t.forward(10)
t.left(270)
t.forward(30)
t.end_fill()

#文字
t.fillcolor('#FFA500')
t.begin_fill()
Skip(t,-340,-15)
t.left(90)
t.forward(20)

t.left(90)
t.forward(190)
t.left(90)
t.forward(20)
t.left(90)
t.forward(190)
t.end_fill()

t.fillcolor('#FFA500')
t.begin_fill()
Skip(t,25,-15)
t.left(90)
t.forward(20)

t.left(90)
t.forward(190)
t.left(90)
t.forward(20)
t.end_fill()

#画框
t.fillcolor('#13e8e8')
t.begin_fill()
Skip(t,-77,-4)
t.left(180)
t.forward(45)
t.left(90)
t.forward(36)
t.left(90)
t.forward(45)
t.left(90)
t.forward(36)
t.end_fill()

t.pencolor("red")
t.pensize(3)
t.fillcolor(1,0,0)
t.begin_fill()
Skip(t,250,-40)
t.left(90)
t.forward(50)
t.left(90)
t.forward(75)
t.left(90)
t.forward(50)
t.left(90)
t.forward(75)
t.end_fill()



t.pencolor("black")

t.fillcolor('#b3afaf')
t.begin_fill()
Skip(t,-100,-350)
t.left(180)
t.forward(100)
t.left(90)
t.forward(20)
t.left(90)
t.forward(100)
t.left(90)
t.forward(20)
t.end_fill()
t.fillcolor('#b3afaf')
t.begin_fill()
Skip(t,-80,-330)
t.left(90)
t.forward(60)
t.left(90)
t.forward(20)
t.left(90)
t.forward(60)
t.left(90)
t.forward(20)
t.end_fill()
t.fillcolor('#b3afaf')
t.begin_fill()
Skip(t,-65,-310)
t.left(90)
t.forward(30)
t.left(90)
t.forward(20)
t.left(90)
t.forward(30)
t.left(90)
t.forward(20)
t.end_fill()

t.pensize(8)
t.pencolor("gray")
Skip(t,-50,-290)
t.left(180)
t.forward(170)
t.pencolor("red")
t.pensize(3)
t.fillcolor(1,0,0)
t.begin_fill()
t.left(-90)
t.forward(75)
t.left(-90)
t.forward(50)
t.left(-90)
t.forward(75)
t.left(-90)
t.forward(50)
t.end_fill()

t.pencolor("yellow")
t.pensize(3)
Skip(t,-37.5,-132.5)
t.fillcolor(1,1,0)
t.begin_fill()
t.left(-90)
for i in range(5):
    t.forward(10)
    t.right(144)
t.end_fill()

Skip(t,-20,-125)
t.pensize(1)
t.fillcolor(1,1,0)
t.begin_fill()
t.left(-90)
for i in range(5):
    t.forward(6)
    t.right(144)
t.end_fill()
Skip(t,-20,-147.5)
t.fillcolor(1,1,0)
t.begin_fill()
t.left(-90)
for i in range(5):
    t.forward(6)
    t.right(144)
t.end_fill()
Skip(t,-15,-133)
t.fillcolor(1,1,0)
t.begin_fill()
t.left(-90)
for i in range(5):
    t.forward(6)
    t.right(144)
t.end_fill()
Skip(t,-15,-140)
t.fillcolor(1,1,0)
t.begin_fill()
t.left(-90)
for i in range(5):
    t.forward(6)
    t.right(144)
t.end_fill()



t.done()
