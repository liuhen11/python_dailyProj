#!/usr/bin/python3

# 导入模块
from tkinter import ttk
import tkinter as tk
import tkinter.font as tkFont
from tkinter import *  # 图形界面库
import tkinter.messagebox as messagebox  # 弹窗
import os
import person_exercise
import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pylab import mpl
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk  # NavigationToolbar2TkAgg
import tkinter as tk


mpl.rcParams['font.sans-serif'] = ['SimHei']  # 中文显示
mpl.rcParams['axes.unicode_minus'] = False  # 负号显示

# 主页面类
class StartPage:
    def __init__(self, parent_window):
        parent_window.destroy()  # 销毁子界面

        self.window = tk.Tk()  # 初始框的声明
        self.window.title('健身会员信息管理系统')
        self.window.geometry('300x470')  # 这里的乘是小x

        label = Label(self.window, text="欢迎您使用健身会员信息管理系统", font=("Verdana", 14))
        label.pack(pady=100)  # pady=100 界面的长度

        Button(self.window, text="管理员登陆", font=tkFont.Font(size=16), command=lambda: AdminPage(self.window), width=30,
               height=2,
               fg='white', bg='gray', activebackground='black', activeforeground='white').pack()
        Button(self.window, text="会员登陆", font=tkFont.Font(size=16), command=lambda: VipPage(self.window), width=30,
               height=2, fg='white', bg='gray', activebackground='black', activeforeground='white').pack()
        Button(self.window, text="关于", font=tkFont.Font(size=16), command=lambda: AboutPage(self.window), width=30,
               height=2,
               fg='white', bg='gray', activebackground='black', activeforeground='white').pack()
        Button(self.window, text='退出系统', height=2, font=tkFont.Font(size=16), width=30, command=self.window.destroy,
               fg='white', bg='gray', activebackground='black', activeforeground='white').pack()

        self.window.mainloop()  # 主消息循环


# 管理员登陆页面类
class AdminPage:
    def __init__(self, parent_window):
        parent_window.destroy()  # 销毁主界面

        self.window = tk.Tk()  # 初始框的声明
        self.window.title('管理员登陆页面')
        self.window.geometry('300x450')  # 这里的乘是小x

        label = tk.Label(self.window, text='管理员登陆', bg='gray', font=('Verdana', 20), width=30, height=2)
        label.pack()

        Label(self.window, text='管理员账号：', font=tkFont.Font(size=14)).pack(pady=25)
        self.admin_username = tk.Entry(self.window, width=25, font=tkFont.Font(size=14), bg='Ivory')
        self.admin_username.pack()

        Label(self.window, text='管理员密码：', font=tkFont.Font(size=14)).pack(pady=25)
        self.admin_pass = tk.Entry(self.window, width=25, font=tkFont.Font(size=14), bg='Ivory', show='*')
        self.admin_pass.pack()

        Button(self.window, text="登陆", width=8, font=tkFont.Font(size=12), command=self.login).pack(pady=40)
        Button(self.window, text="返回首页", width=8, font=tkFont.Font(size=12), command=self.back).pack()

        self.window.protocol("WM_DELETE_WINDOW", self.back)  # 捕捉右上角关闭点击
        self.window.mainloop()  # 进入消息循环

    def login(self):
        print(str(self.admin_username.get()))
        print(str(self.admin_pass.get()))
        admin_pass = None

        f = open('./yonghu.txt', 'r')  # 这个是管理员账号密码数据库
        login_info = f.readline()
        print(login_info)
        admin_id = login_info.split(',')[0]
        admin_pass = login_info.split(',')[1]
            # 打印结果
        print("admin_id=%s,admin_pass=%s" % (admin_id, admin_pass))
        f.close()

        print("正在登陆管理员管理界面")
        print("self", self.admin_pass.get())
        print("local", admin_pass)

        if self.admin_pass.get() == admin_pass and self.admin_username.get() == admin_id:  # 判断账号和密码输入是否正确
            AdminManage(self.window)  # 进入管理员操作界面
        else:
            messagebox.showinfo('警告！', '用户名或密码不正确！')

    def back(self):
        StartPage(self.window)  # 显示主窗口 销毁本窗口


# 会员登陆页面  保留接口，这个是会员自己登录的页面
class VipPage:
    def __init__(self, parent_window):
        parent_window.destroy()  # 销毁主界面

        self.window = tk.Tk()  # 初始框的声明
        self.window.title('会员登陆')
        self.window.geometry('300x450')  # 这里的乘是小x

        label = tk.Label(self.window, text='会员登陆', bg='green', font=('Verdana', 20), width=30, height=2)
        label.pack()

        Label(self.window, text='会员账号：', font=tkFont.Font(size=14)).pack(pady=25)
        self.vip_id = tk.Entry(self.window, width=30, font=tkFont.Font(size=14), bg='Ivory')
        self.vip_id.pack()

        Label(self.window, text='会员密码：', font=tkFont.Font(size=14)).pack(pady=25)
        self.vip_pass = tk.Entry(self.window, width=30, font=tkFont.Font(size=14), bg='Ivory', show='*')
        self.vip_pass.pack()

        Button(self.window, text="登陆", width=8, font=tkFont.Font(size=12), command=self.login).pack(pady=40)
        Button(self.window, text="返回首页", width=8, font=tkFont.Font(size=12), command=self.back).pack()

        self.window.protocol("WM_DELETE_WINDOW", self.back)  # 捕捉右上角关闭点击
        self.window.mainloop()  # 进入消息循环

    def login(self):       # 会员员登录函数声明
        print(str(self.vip_id.get()))
        print(str(self.vip_pass.get()))
        vip_pass = None

        print("正在登陆会员信息查看界面")
        print("self", self.vip_pass.get())
        print("local", vip_pass)

        if self.vip_pass.get() == vip_pass:               # 判断密码是否正确
            VipView(self.window, self.vip_id.get())  # 进入会员信息查看界面
        else:
            messagebox.showinfo('警告！', '用户名或密码不正确！')

    def back(self):
        StartPage(self.window)  # 显示主窗口 销毁本窗口


# 管理员操作界面类
class AdminManage:
    def __init__(self, parent_window):
        parent_window.destroy()  # 销毁主界面

        self.window = Tk()  # 初始框的声明
        self.window.title('管理员操作界面')

        self.frame_left_top = tk.Frame(width=300, height=300)    # 定义该页面功能部件放置区域
        self.frame_right_top = tk.Frame(width=200, height=300)    # 定义该页面功能部件放置区域
        self.frame_center = tk.Frame(width=560, height=400)      # 定义该页面功能部件放置区域
        self.frame_bottom = tk.Frame(width=650, height=50)          # 定义该页面功能部件放置区域

        # 定义下方中心列表区域
        self.columns = ("编号", "姓名", "密码", "性别","生日","会员等级","积分")   # 设置列名
        self.tree = ttk.Treeview(self.frame_center, show="headings", height=18, columns=self.columns)      # 定义会员信息表格
        self.vbar = ttk.Scrollbar(self.frame_center, orient=VERTICAL, command=self.tree.yview)            # 滚动条显示
        # 定义树形结构与滚动条
        self.tree.configure(yscrollcommand=self.vbar.set)

        # 表格的标题
        self.tree.column("编号", width=50, anchor='center')  # 表示列,不显示
        self.tree.column("姓名", width=80, anchor='center')
        self.tree.column("密码", width=110, anchor='center')
        self.tree.column("性别", width=80, anchor='center')
        self.tree.column("生日", width=80, anchor='center')
        self.tree.column("会员等级", width=80, anchor='center')
        self.tree.column("积分", width=80, anchor='center')

        # 调用方法获取表格内容插入
        self.tree.grid(row=0, column=0, sticky=NSEW)
        self.vbar.grid(row=0, column=1, sticky=NS)

        self.id = []             # 会员编号
        self.name = []           # 会员姓名
        self.passcode = []       # 会员密码
        self.gender = []         # 会员性别
        self.birth = []          # 会员生日
        self.level = []          # 会员等级
        self.score = []          # 会员积分

        with open('huiyuaninfo.txt','r',encoding='gb2312') as f:
             huiyuaninfo =  f.readlines()
             for row in huiyuaninfo:
                self.id.append(row.split(',')[0])
                self.name.append(row.split(',')[1])
                self.passcode.append(row.split(',')[2])
                self.gender.append(row.split(',')[3])
                self.birth.append(row.split(',')[4])
                self.level.append(row.split(',')[5])
                self.score.append(row.split(',')[6])

                print(self.id)
                print(self.name)
                print(self.passcode)
                print(self.gender)
                print(self.birth)
                print(self.level)
                print(self.score)

        print("test***********************")
        for i in range(min(len(self.id), len(self.name),len(self.passcode), len(self.gender), len(self.birth),len(self.level),len(self.score))):  # 往表格写入数据
            self.tree.insert('', i, values=(self.id[i], self.name[i],self.passcode[i] ,self.gender[i], self.birth[i],self.level[i],self.score[i]))

        for col in self.columns:  # 绑定函数，使表头可排序 即根据列名排序
            self.tree.heading(col, text=col,
                              command=lambda _col=col: self.tree_sort_column(self.tree, _col, False))

        # 定义顶部区域
        # 定义左上方区域
        self.top_title = Label(self.frame_left_top, text="会员信息:", font=('Verdana', 20))
        self.top_title.grid(row=0, column=0, columnspan=2, sticky=NSEW, padx=50, pady=10)

        self.left_top_frame = tk.Frame(self.frame_left_top)
        self.var_id = StringVar()  # 声明编号
        self.var_name = StringVar()  # 声明姓名
        self.var_passcode = StringVar()  # 声明密码
        self.var_gender = StringVar()  # 声明性别
        self.var_birth = StringVar()    # 声明出生日期
        self.var_level = StringVar()   #声明会员等级
        self.var_score = StringVar()   # 声明积分

        # 编号
        # 并获取文本框中内容
        self.right_top_id_label = Label(self.frame_left_top, text="编号：", font=('Verdana', 15))
        self.right_top_id_entry = Entry(self.frame_left_top, textvariable=self.var_id, font=('Verdana', 15))
        self.right_top_id_label.grid(row=1, column=0)  # 位置设置
        self.right_top_id_entry.grid(row=1, column=1)
        # 姓名
        self.right_top_name_label = Label(self.frame_left_top, text="姓名：", font=('Verdana', 15))
        self.right_top_name_entry = Entry(self.frame_left_top, textvariable=self.var_name, font=('Verdana', 15))
        self.right_top_name_label.grid(row=2, column=0)  # 位置设置
        self.right_top_name_entry.grid(row=2, column=1)
        # 密码
        self.right_top_name_label = Label(self.frame_left_top, text="密码：", font=('Verdana', 15))
        self.right_top_name_entry = Entry(self.frame_left_top, textvariable=self.var_passcode, font=('Verdana', 15))
        self.right_top_name_label.grid(row=3, column=0)  # 位置设置
        self.right_top_name_entry.grid(row=3, column=1)
        # 性别
        self.right_top_gender_label = Label(self.frame_left_top, text="性别：", font=('Verdana', 15))
        self.right_top_gender_entry = Entry(self.frame_left_top, textvariable=self.var_gender,font=('Verdana', 15))
        self.right_top_gender_label.grid(row=4, column=0)  # 位置设置
        self.right_top_gender_entry.grid(row=4, column=1)
        # 生日
        self.right_top_gender_label = Label(self.frame_left_top, text="生日：", font=('Verdana', 15))
        self.right_top_gender_entry = Entry(self.frame_left_top, textvariable=self.var_birth,font=('Verdana', 15))
        self.right_top_gender_label.grid(row=5, column=0)  # 位置设置
        self.right_top_gender_entry.grid(row=5, column=1)
        # 会员等级
        self.right_top_gender_label = Label(self.frame_left_top, text="会员等级：", font=('Verdana', 15))
        self.right_top_gender_entry = Entry(self.frame_left_top, textvariable=self.var_level, font=('Verdana', 15))
        self.right_top_gender_label.grid(row=6, column=0)  # 位置设置
        self.right_top_gender_entry.grid(row=6, column=1)
        # 积分
        self.right_top_gender_label = Label(self.frame_left_top, text="积分：", font=('Verdana', 15))
        self.right_top_gender_entry = Entry(self.frame_left_top, textvariable=self.var_score, font=('Verdana', 15))
        self.right_top_gender_label.grid(row=7, column=0)  # 位置设置
        self.right_top_gender_entry.grid(row=7, column=1)

        # 定义右上方区域
        self.right_top_title = Label(self.frame_right_top, text="操作：", font=('Verdana', 20))

        self.tree.bind('<Button-1>', self.click)  # 左键获取位置
        self.right_top_button1 = ttk.Button(self.frame_right_top, text='添加会员信息', width=20, command=self.new_row)
        self.right_top_button2 = ttk.Button(self.frame_right_top, text='修改选中会员信息', width=20,
                                            command=self.updata_row)
        self.right_top_button3 = ttk.Button(self.frame_right_top, text='删除选中会员信息', width=20,
                                            command=self.del_row)
        self.right_top_button4 = ttk.Button(self.frame_right_top, text='查询会员信息', width=20,
                                            command=self.search_row)
        self.right_top_button5 = ttk.Button(self.frame_right_top, text='会员等级统计升级', width=20,
                                            command=self.static_user)
        self.right_top_button6 = ttk.Button(self.frame_bottom, text='密码修改', width=20,
                                            command=self.password_modify)

        # 位置设置
        self.right_top_title.grid(row=1, column=0, sticky=NSEW,padx = 70, pady=10)
        self.right_top_button1.grid(row=2, column=0, padx=20, pady=9)
        self.right_top_button2.grid(row=3, column=0, padx=20, pady=9)
        self.right_top_button3.grid(row=4, column=0, padx=20, pady=9)
        self.right_top_button4.grid(row=5, column=0, padx=20, pady=9)
        self.right_top_button5.grid(row=6, column=0, padx=20, pady=9)
        self.right_top_button6.grid(row=0, column=0)

        # 整体区域定位
        self.frame_left_top.grid(row=0, column=0, padx=2, pady=10)    # 左上区域
        self.frame_right_top.grid(row=0, column=1, padx=30, pady=30)  # 右上区域
        self.frame_center.grid(row=1, column=0, columnspan=2, padx=4, pady=5) # 中心区域即表格区域
        self.frame_bottom.grid(row=2, column=0, columnspan=2)         # 底部区域

        self.frame_left_top.grid_propagate(0)
        self.frame_right_top.grid_propagate(0)
        self.frame_center.grid_propagate(0)
        self.frame_bottom.grid_propagate(0)

        self.frame_left_top.tkraise()  # 开始显示主菜单
        self.frame_right_top.tkraise()  # 开始显示主菜单
        self.frame_center.tkraise()  # 开始显示主菜单
        self.frame_bottom.tkraise()  # 开始显示主菜单

        self.window.protocol("WM_DELETE_WINDOW", self.back)  # 捕捉右上角关闭点击
        self.window.mainloop()  # 进入消息循环

    def back(self):
        StartPage(self.window)  # 显示主窗口 销毁本窗口

    def click(self, event):      # 点击事件
        self.col = self.tree.identify_column(event.x)  # 列
        self.row = self.tree.identify_row(event.y)  # 行

        print(self.col)
        print(self.row)
        self.row_info = self.tree.item(self.row, "values")
        print(self.row_info[0])
        self.var_id.set(self.row_info[0])
        self.var_name.set(self.row_info[1])
        self.var_passcode.set(self.row_info[2])
        self.var_gender.set(self.row_info[3])
        self.var_birth.set(self.row_info[4])
        self.var_level.set(self.row_info[5])
        self.var_score.set(self.row_info[6])
        self.right_top_id_entry = Entry(self.frame_left_top, state='disabled', textvariable=self.var_id,
                                        font=('Verdana', 15))

        print('')
    # 按照编号和积分大小排序
    def tree_sort_column(self, tv, col, reverse):  # Treeview、列名、排列方式
        print(col)
        if col == '编号' or '积分':
            l = [(int(tv.set(k, col)), k) for k in tv.get_children('')]
            print(l)
            l.sort(reverse=reverse)  # 排序方式
            print(l)
            # rearrange items in sorted positions
            for index, (val, k) in enumerate(l):  # 根据排序后索引移动
                tv.move(k, '', index)
            tv.heading(col, command=lambda: self.tree_sort_column(tv, col, not reverse))  # 重写标题，使之成为再点倒序的标题
    # 定义会员信息添加函数
    def new_row(self):
        print('123')
        print(self.var_id.get())
        print(self.id)
        if str(self.var_id.get()) in self.id:
            messagebox.showinfo('警告！', '该会员信息已存在！')
        else:
            if self.var_id.get() != '' and self.var_name.get() != '' and self.var_passcode.get() != ''and self.var_gender.get() != ''and self.var_birth.get() != ''and self.var_level.get() != ''and self.var_score.get() != '':

                per = person_exercise.Person(self.var_id.get(),self.var_name.get(), self.var_passcode.get() ,self.var_gender.get(), self.var_birth.get(), \
                                             self.var_level.get(),self.var_score.get())
                info_txt = self.var_id.get() +','+ self.var_name.get()+','+ self.var_passcode.get() + ',' + self.var_gender.get()+ ',' + self.var_birth.get() \
                                             + ','+ self.var_level.get() + ',' + self.var_score.get()
                try:
                    try:
                        huiyuaninfo_txt = open('huiyuaninfo.txt', 'a',encoding='gb2312')     # 读取会员信息数据库
                    except Exception as e:
                        huiyuaninfo_txt = open('huiyuaninfo.txt', 'w',encoding='gb2312')     # 将会员信息写入数据库
                    huiyuaninfo_txt.write(str(info_txt)+'\n')
                    huiyuaninfo_txt.close()
                except:

                    messagebox.showinfo('警告！', '本地数据库连接失败！')

                self.id.append(self.var_id.get())   # 将文本框中编号信息添加进编号列表
                self.name.append(self.var_name.get()) # 将文本框中姓名信息添加进姓名列表
                self.passcode.append(self.var_passcode.get()) # 将文本框中密码信息添加进密码列表
                self.gender.append(self.var_gender.get())     # 将文本框中性别信息添加进性别列表
                self.birth.append(self.var_birth.get())      # 将文本框中生日信息添加进生日列表
                self.level.append(self.var_level.get())       # 将文本框中会员等级信息添加进等级列表
                self.score.append(self.var_score.get())      # 将文本框中积分信息添加进积分列表
                self.tree.insert('', len(self.id) - 1, values=(
                    self.id[len(self.id) - 1], self.name[len(self.id) - 1], self.passcode[len(self.id) - 1], self.gender[len(self.id) - 1],  self.birth[len(self.id) - 1],
                    self.level[len(self.id) - 1], self.score[len(self.id) - 1]))         # 将上述信息添加进表格
                self.tree.update()                                                       # 表格更新
                messagebox.showinfo('提示！', '插入成功！')
            else:
                messagebox.showinfo('警告！', '请填写会员信息数据')

    # 定义会员信息修改函数
    def updata_row(self):
        res = messagebox.askyesnocancel('警告！', '是否更新所填数据？')
        if res == True:
            if self.var_id.get() == self.row_info[0]:  # 如果所填编号 与 所选编号一致
                info_txt1 = self.var_id.get() + ',' + self.var_name.get() + ',' + self.var_passcode.get() + ',' + self.var_gender.get() + ',' + self.var_birth.get() \
                           + ',' + self.var_level.get() + ',' + self.var_score.get().strip()
                print(info_txt1)
                try:
                    with open('huiyuaninfo.txt','r',encoding='gb2312') as ft:   # 将从文本框修改的信息添加进本地txt数据库
                        modify_infos = ft.readlines()
                        for modify_info_index in range(len(modify_infos)):
                            if modify_infos[modify_info_index].split(',')[0] == self.var_id.get():
                                modify_infos[modify_info_index] = info_txt1 + '\n'
                    print(modify_infos)
                    with open('huiyuaninfo.txt','w', encoding='gb2312') as ft:
                        ft.writelines(modify_infos)
                    messagebox.showinfo('提示！', '更新成功！')
                except:
                    messagebox.showinfo('警告！', '更新失败，本地数据库连接失败！')

                id_index = self.id.index(self.row_info[0])
                self.name[id_index] = self.var_name.get()
                self.passcode[id_index] = self.var_passcode.get()
                self.gender[id_index] = self.var_gender.get()
                self.birth[id_index] = self.var_birth.get()
                self.level[id_index] = self.var_level.get()
                self.score[id_index] = self.var_score.get()

                self.tree.item(self.tree.selection()[0], values=(
                    self.var_id.get(), self.var_name.get(), self.var_passcode.get(), self.var_gender.get(),self.var_birth.get(),self.var_level.get(),
                    self.var_score.get()))  # 修改对于行信息
            else:
                messagebox.showinfo('警告！', '不能修改会员编号！')

    # 定义会员信息删除函数
    def del_row(self):
        res = messagebox.askyesnocancel('警告！', '是否删除所选数据？')
        if res == True:
            print(self.row_info[0])  # 鼠标选中的编号
            print(self.tree.selection()[0])  # 行号
            print(self.tree.get_children())  # 所有行

            try:
                with open('huiyuaninfo.txt','r', encoding='gb2312') as f:
                    lines = f.readlines()
                    # print(lines)
                with open('huiyuaninfo.txt','w', encoding='gb2312') as f_w:
                    for line in lines:
                        if self.var_id.get() == line.split(',')[0]:
                            continue
                        f_w.write(line)

                messagebox.showinfo('提示！', '删除成功！')
            except:
                messagebox.showinfo('警告！', '删除失败，数据库连接失败！')

            id_index = self.id.index(self.row_info[0])
            print(id_index)
            del self.id[id_index]
            del self.name[id_index]
            del self.passcode[id_index]
            del self.gender[id_index]
            del self.birth[id_index]
            del self.level[id_index]
            del self.score[id_index]
            print(self.id)
            self.tree.delete(self.tree.selection()[0])  # 删除所选行
            print(self.tree.get_children())

 # 定义会员信息 按编号或者姓名或者按会员等级信息查询会员信息
    def search_row(self):
        search_id = self.var_id.get()      # 获取要查询的id
        search_name = self.var_name.get()  # 获取要查询的姓名
        search_level = self.var_level.get() # 获取要查询的等级
        for i in self.tree.get_children():  # 首先删除表格信息
            self.tree.delete(i)
        person_info = open('huiyuaninfo.txt','r',encoding='gb2312')  # 打开本地txt数据库
        person_infos = person_info.readlines()
        c = 0
        for i in range(len(person_infos)):
            self.id.append(person_infos[i].split(',')[0])
            self.name.append(person_infos[i].split(',')[1])
            self.passcode.append(person_infos[i].split(',')[2])
            self.gender.append(person_infos[i].split(',')[3])
            self.birth.append(person_infos[i].split(',')[4])
            self.level.append(person_infos[i].split(',')[5])
            self.score.append(person_infos[i].split(',')[6])
            if search_id!="" and search_id == person_infos[i].split(',')[0]:   #搜选id并显示在表格
                self.tree.insert('', 0, values=(
                self.id[i], self.name[i], self.passcode[i], self.gender[i], self.birth[i], self.level[i],
                self.score[i]))

            elif search_name != "" and search_name == person_infos[i].split(',')[1]:  # 搜选姓名并显示在表格
                self.tree.insert('', 0, values=(
                    self.id[i], self.name[i], self.passcode[i], self.gender[i], self.birth[i], self.level[i],
                    self.score[i]))
            elif search_level != "" and search_level == person_infos[i].split(',')[5]:  # 按等级查找显示在表格
                self.tree.insert('', c, values=(
                    self.id[i], self.name[i], self.passcode[i], self.gender[i], self.birth[i], self.level[i],
                    self.score[i]))
                c = c + 1
            elif self.var_id.get() == '' and self.var_name.get() == '' and self.var_passcode.get() == ''and self.var_gender.get() == ''and self.var_birth.get() == ''and self.var_level.get() == ''and self.var_score.get() == '':
                self.tree.insert('', i, values=(
                    self.id[i], self.name[i], self.passcode[i], self.gender[i], self.birth[i], self.level[i],
                    self.score[i]))   # 显示所有信息
    def static_user(self):  # 定义会员等级统计升级函数
        self.rowx = self.tree.get_children()
        list_level = []
        for i in self.rowx:
            print(self.tree.item(i, 'values')[5])
            list_level.append(self.tree.item(i, 'values')[5])
        form = From(list_level)

    def password_modify(self):
        xc = Passmodify(self.window)

# 会员查看信息界面
class VipView:
    def __init__(self, parent_window, student_id):
        parent_window.destroy()  # 销毁主界面

        self.window = tk.Tk()  # 初始框的声明
        self.window.title('关于')
        self.window.geometry('300x450')  # 这里的乘是小x

        label = tk.Label(self.window, text='会员信息查看', bg='green', font=('Verdana', 20), width=30, height=2)
        label.pack(pady=20)

        self.id = '编号:' + ''
        self.name = '姓名:' + ''
        self.passcode = '密码:' + ''
        self.gender = '性别:' + ''
        self.birth = '生日:' + ''
        self.level = '会员等级:' + ''
        self.score = '积分:' + ''

        Label(self.window, text=self.id, font=('Verdana', 18)).pack(pady=5)
        Label(self.window, text=self.name, font=('Verdana', 18)).pack(pady=5)
        Label(self.window, text=self.passcode, font=('Verdana', 18)).pack(pady=5)
        Label(self.window, text=self.gender, font=('Verdana', 18)).pack(pady=5)
        Label(self.window, text=self.birth, font=('Verdana', 18)).pack(pady=5)
        Label(self.window, text=self.level, font=('Verdana', 18)).pack(pady=5)
        Label(self.window, text=self.score, font=('Verdana', 18)).pack(pady=5)

        Button(self.window, text="返回首页", width=8, font=tkFont.Font(size=16), command=self.back).pack(pady=25)

        self.window.protocol("WM_DELETE_WINDOW", self.back)  # 捕捉右上角关闭点击
        self.window.mainloop()  # 进入消息循环

    def back(self):
        StartPage(self.window)  # 显示主窗口 销毁本窗口


# About页面
class AboutPage:
    def __init__(self, parent_window):
        parent_window.destroy()  # 销毁主界面

        self.window = tk.Tk()  # 初始框的声明
        self.window.title('关于')
        self.window.geometry('300x450')  # 这里的乘是小x

        label = tk.Label(self.window, text='会员信息管理系统', bg='gray', font=('Verdana', 20), width=30, height=2)
        label.pack()

        Label(self.window, text='作者：袁凡和不语', font=('Verdana', 18)).pack(pady=30)
        Label(self.window, text='xxxxxx大学', font=('Verdana', 18)).pack(pady=5)

        Button(self.window, text="返回首页", width=8, font=tkFont.Font(size=12), command=self.back).pack(pady=100)

        self.window.protocol("WM_DELETE_WINDOW", self.back)  # 捕捉右上角关闭点击
        self.window.mainloop()  # 进入消息循环

    def back(self):
        StartPage(self.window)  # 显示主窗口 销毁本窗口

# 会员等级统计界面类
class From:
    def __init__(self,list_level):
        self.root = tk.Tk()  # 创建主窗体
        self.canvas = tk.Canvas()  # 创建一块显示图形的画布
        self.list_level = list_level
        self.figure = self.create_matplotlib(self.list_level)  # 返回matplotlib所画图形的figure对象
        self.create_form(self.figure)  # 将figure显示在tkinter窗体上面
        self.root.mainloop()

    def create_matplotlib(self,x):
        # 创建绘图对象f
        f = plt.figure(num=2, figsize=(10, 8), dpi=80, facecolor="pink", edgecolor='green', frameon=True)
        # 创建一副子图
        fig1 = plt.subplot(1, 1, 1)

        x_single = set(x)
        results_x = []
        results_y = []
        for i in x_single:
            results_x.append(i)
            results_y.append(x.count(i))

        fig1.bar(x=results_x,height=results_y,label='会员等级数目统计',color='steelblue',alpha=0.8)
        for xx,yy in zip(results_x,results_y):
            plt.text(xx,yy,str(yy),ha='center',va='bottom',fontsize=20,rotation=0)
        plt.title('会员等级具体情况统计柱状图',fontsize=34)
        plt.xlabel('会员等级',fontsize=24)
        plt.ylabel('人数',fontsize=24)

        return f

    def create_form(self, figure):
        # 把绘制的图形显示到tkinter窗口上
        self.canvas = FigureCanvasTkAgg(figure, self.root)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        # 把matplotlib绘制图形的导航工具栏显示到tkinter窗口上
        toolbar = NavigationToolbar2Tk(self.canvas,
                                       self.root)  # matplotlib 2.2版本之后推荐使用NavigationToolbar2Tk，若使用NavigationToolbar2TkAgg会警告
        toolbar.update()
        self.canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=1)

# 进入密码修改页面类
class Passmodify:
    def __init__(self, parent_window):
        parent_window.destroy()  # 销毁主界面

        self.window = tk.Tk()  # 初始框的声明
        self.window.title('关于')
        self.window.geometry('100x200')  # 这里的乘是小x

        label = tk.Label(self.window, text='请输入新密码', bg='gray', font=('Verdana', 14), width=25, height=2)
        label.pack()
        self.new_passwprd = tk.Entry(self.window, width=25, font=tkFont.Font(size=14), bg='Ivory')
        self.new_passwprd.pack()
        Button(self.window, text="确认", width=8, font=tkFont.Font(size=12), command=self.password_modfy).pack(pady=40)
        self.window.protocol("WM_DELETE_WINDOW", self.back)  # 捕捉右上角关闭点击
        self.window.mainloop()  # 进入消息循环

    def back(self):
        AdminManage(self.window)  # 显示主窗口 销毁本窗口

    def password_modfy(self):
        res = messagebox.askyesnocancel('警告！', '是否确认修改？')
        if res == True:
            try:
                with open('yonghu.txt', 'r') as ft:
                    modify_prdinfos = ft.readlines()
                    modify_prdinfos[0] = modify_prdinfos[0].split(',')[0] + ',' + self.new_passwprd.get()
                print(modify_prdinfos)
                with open('yonghu.txt', 'w') as ft:
                    ft.writelines(modify_prdinfos)
                # #     cursor.execute(sql)  # 执行sql语句
                # #     db.commit()  # 提交到数据库执行
                messagebox.showinfo('提示！', '修改成功！请重新登录')
                StartPage(self.window)
            except:
                #     db.rollback()  # 发生错误时回滚
                messagebox.showinfo('警告！', '更新失败，本地txt数据库连接失败！')

        else:
            AdminManage(self.window)

if __name__ == '__main__':
    try:
        # 初始化TK主界面窗口
        window = tk.Tk()
        # 进入开始页面
        StartPage(window)
    except:  # 出现异常，比如本地txt文件不存在时，会报错
        messagebox.showinfo('错误！', '连接本地txt数据库失败！')
