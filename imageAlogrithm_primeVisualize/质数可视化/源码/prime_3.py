import turtle
from math import sqrt

# 判断是否为素数
def is_prime(num):
    for i in range(2, round(sqrt(num))+1):
        # 能整除，非素数
        if num % i == 0:
            return False
    return True

# %%
# 定义画布的大小和背景
turtle.setup(0.8, 0.9)
turtle.screensize(100, 100, 'black')
minxv = -50 # 定义起始位置
minyv = -80
maxxv = turtle.window_width()+minxv
maxyv = turtle.window_height()+minyv
turtle.setworldcoordinates(minxv, minyv, maxxv, maxyv)
# 定义画笔的速度
turtle.speed(0)
turtle.Turtle().screen.delay(0)
# 绘制图形的宽度
turtle.pencolor("white")
turtle.pensize(1)
turtle.setheading(-90)
turtle.hideturtle()

# %%
num = 0
turtle.pendown()
run_flag = True
path = "out.txt"
with open(path,"w",encoding="utf-8") as f:
    while run_flag:
        turtle.forward(1)
        num = num+1

        xcor=round(turtle.xcor())
        ycor=round(turtle.ycor())

        # 转向
        if is_prime(num):
            turtle.right(90) # 一直右转90

            keyIn=(xcor,ycor)
            print("{0},-->{1}".format(keyIn,num))
            f.write("{0},-->{1}".format(keyIn,num))

        # 停止
        if xcor > maxxv or xcor < minxv or ycor > maxyv or ycor < minyv:
            run_flag = False

# 保存图像
ts = turtle.getscreen()
ts.getcanvas().postscript(file="work.eps")

# 不关闭窗口
turtle.mainloop()
