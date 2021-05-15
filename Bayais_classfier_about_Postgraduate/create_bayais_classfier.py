# !/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time : 2021/4/9 15:56
# @Author : Mat
# @Email : mat_wu@163.com
# @File : create_bayais_classfier.py
# @Software: PyCharm

import  pandas as pd

graduate_data = pd.read_excel('新冠疫情对备考研究生心理影响分析_517_517.xlsx', sheet_name='最终数据', index_col='序号', header=0)
print(graduate_data)

# 统计参加问卷调查的总人数
graduate_total = len(graduate_data)
print(graduate_total)

# 分别统计坚持考研和放弃考研的人数，并计算出坚持考研和放弃考研的概率，此概率为先验概率
# 因为属性均为离散的 0 1 2值，且均在训练问卷样本中出现，所以不需要进行拉普拉斯修正
graduate_data_stick_to =  graduate_data[graduate_data['新冠疫情期间，您是否坚持考研到最后（0=是）'] == 0]   # 筛选出坚持考研的人数属性信息
graduate_total_stcik_to = len(graduate_data_stick_to)
print(graduate_total_stcik_to)
graduate_data_give_up = graduate_data[graduate_data['新冠疫情期间，您是否坚持考研到最后（0=是）'] == 1]     # 筛选出放弃考研的人数属性信息
graduate_total_give_up = len(graduate_data_give_up)
print(graduate_total_give_up)

# 计算出坚持和放弃考研的概率，根据上述频数计算
p_graduate_total_stcik_to = graduate_total_stcik_to / (graduate_total_stcik_to + graduate_total_give_up)
print(p_graduate_total_stcik_to)
p_graduate_total_give_up  = graduate_total_give_up /  (graduate_total_stcik_to + graduate_total_give_up)
print(p_graduate_total_give_up)

# bayais 假设各属性特征条件独立同分布，即相互之间具有条件独立性，各自不影响
# 统计各属性条件分别在坚持考研和放弃考研中的人数以及概率

# 统计性别在坚持考研和放弃中的人数及概率
# 计算在坚持考研和放弃考研的条件下该学生是男生的人数和概率
# 人数统计
boy_total_in_stick_to = len(graduate_data_stick_to[graduate_data_stick_to['性别（0=男）'] == 0])
print(boy_total_in_stick_to)
boy_total_in_give_up = len(graduate_data_give_up[graduate_data_give_up['性别（0=男）'] == 0])
print(boy_total_in_give_up)
# 概率计算
p_boy_in_stick_to = boy_total_in_stick_to / graduate_total_stcik_to
print(p_boy_in_stick_to)
p_boy_in_give_up = boy_total_in_give_up / graduate_total_give_up
print(p_boy_in_give_up)

# 人数统计
girl_total_in_stick_to = len(graduate_data_stick_to[graduate_data_stick_to['性别（0=男）'] == 1])
print(girl_total_in_stick_to)
girl_total_in_give_up = len(graduate_data_give_up[graduate_data_give_up['性别（0=男）'] == 1])
print(girl_total_in_give_up)
# 概率计算
p_girl_in_stick_to = girl_total_in_stick_to / graduate_total_stcik_to
print(p_girl_in_stick_to)
p_girl_in_give_up = girl_total_in_give_up / graduate_total_give_up
print(p_girl_in_give_up)

# 统计分别在坚持和放弃考研的条件下新冠疫情是否让学生焦虑的人数和概率
# 人数统计
is_anxious_in_stick_to = len(graduate_data_stick_to[graduate_data_stick_to['新冠疫情是否让你焦虑（0=是）'] == 0])
print(is_anxious_in_stick_to)
is_anxious_in_give_up = len(graduate_data_give_up[graduate_data_give_up['新冠疫情是否让你焦虑（0=是）'] == 0])
print(is_anxious_in_give_up)
# 概率计算
p_is_anxious_in_stick_to = is_anxious_in_stick_to / graduate_total_stcik_to
print(p_is_anxious_in_stick_to)
p_is_anxious_in_give_up = is_anxious_in_give_up / graduate_total_give_up
print(p_is_anxious_in_give_up)

# 人数统计
no_anxious_in_stick_to = len(graduate_data_stick_to[graduate_data_stick_to['新冠疫情是否让你焦虑（0=是）'] == 1])
print(no_anxious_in_stick_to)
no_anxious_in_give_up = len(graduate_data_give_up[graduate_data_give_up['新冠疫情是否让你焦虑（0=是）'] == 1])
print(no_anxious_in_give_up)
# 概率计算
p_no_anxious_in_stick_to = no_anxious_in_stick_to / graduate_total_stcik_to
print(p_no_anxious_in_stick_to)
p_no_anxious_in_give_up = no_anxious_in_give_up / graduate_total_give_up
print(p_no_anxious_in_give_up)

# 统计和计算分别在坚持和放弃考研的条件下消极和积极态度的人数和概率
# 人数统计
negtive_in_stick_to = len(graduate_data_stick_to[graduate_data_stick_to['消极or积极or二者之间（0=消极）'] == 0])
print(negtive_in_stick_to)
negtive_in_give_up = len(graduate_data_give_up[graduate_data_give_up['消极or积极or二者之间（0=消极）'] == 0])
print(negtive_in_give_up)
# 概率计算
p_negtive_in_stick_to = negtive_in_stick_to / graduate_total_stcik_to
print(p_negtive_in_stick_to)
p_negtive_in_give_up = negtive_in_give_up / graduate_total_give_up
print(p_negtive_in_give_up)

# 人数统计
postive_in_stick_to = len(graduate_data_stick_to[graduate_data_stick_to['消极or积极or二者之间（0=消极）'] == 1])
print(postive_in_stick_to)
positive_in_give_up = len(graduate_data_give_up[graduate_data_give_up['消极or积极or二者之间（0=消极）'] == 1])
print(positive_in_give_up)
# 概率计算
p_postive_in_stick_to = postive_in_stick_to / graduate_total_stcik_to
print(p_postive_in_stick_to)
p_positive_in_give_up = positive_in_give_up / graduate_total_give_up
print(p_positive_in_give_up)

# 态度介于二者之间
# 人数统计
between_in_stick_to = len(graduate_data_stick_to[graduate_data_stick_to['消极or积极or二者之间（0=消极）'] == 2])
print(between_in_stick_to)
between_in_give_up = len(graduate_data_give_up[graduate_data_give_up['消极or积极or二者之间（0=消极）'] == 2])
print(between_in_give_up)
# 概率计算
p_between_in_stick_to = between_in_stick_to / graduate_total_stcik_to
print(p_between_in_stick_to)
p_between_in_give_up = between_in_give_up / graduate_total_give_up
print(p_between_in_give_up)

# 统计和计算分别在坚持和放弃考研的条件下新冠疫情让家庭财产损失与否的概率
# 人数统计
is_loss_in_stick_to = len(graduate_data_stick_to[graduate_data_stick_to['新冠疫情让您的家庭收入产生损失（0=是）'] == 0])
print(is_loss_in_stick_to)
is_loss_in_give_up = len(graduate_data_give_up[graduate_data_give_up['新冠疫情让您的家庭收入产生损失（0=是）'] == 0])
print(is_loss_in_give_up)
# 概率计算
p_is_loss_in_stick_to = is_loss_in_stick_to / graduate_total_stcik_to
print(p_is_loss_in_stick_to)
p_is_loss_in_give_up = is_loss_in_give_up / graduate_total_give_up
print(p_is_loss_in_give_up)

# 人数统计
no_loss_in_stick_to = len(graduate_data_stick_to[graduate_data_stick_to['新冠疫情让您的家庭收入产生损失（0=是）'] == 1])
print(no_loss_in_stick_to)
no_loss_in_give_up = len(graduate_data_give_up[graduate_data_give_up['新冠疫情让您的家庭收入产生损失（0=是）'] == 1])
print(no_loss_in_give_up)
# 概率计算
p_no_loss_in_stick_to = no_loss_in_stick_to / graduate_total_stcik_to
print(p_no_loss_in_stick_to)
p_no_loss_in_give_up = no_loss_in_give_up / graduate_total_give_up
print(p_no_loss_in_give_up)

# 统计和计算 分别在坚持和放弃考研的条件下学校招生是否扩大的人数和概率
# 人数统计
enlarge_in_stick_to = len(graduate_data_stick_to[graduate_data_stick_to['新冠疫情使您的报考学校扩大招生（0=是）'] == 0])
print(enlarge_in_stick_to)
enlarge_in_give_up = len(graduate_data_give_up[graduate_data_give_up['新冠疫情使您的报考学校扩大招生（0=是）'] == 0])
print(enlarge_in_give_up)
# 概率计算
p_enlarge_in_stick_to = enlarge_in_stick_to / graduate_total_stcik_to
print(p_enlarge_in_stick_to)
p_enlarge_in_give_up = enlarge_in_give_up / graduate_total_give_up
print(p_enlarge_in_give_up)

# 人数统计
noenlarge_in_stick_to = len(graduate_data_stick_to[graduate_data_stick_to['新冠疫情使您的报考学校扩大招生（0=是）'] == 1])
print(noenlarge_in_stick_to)
noenlarge_in_give_up = len(graduate_data_give_up[graduate_data_give_up['新冠疫情使您的报考学校扩大招生（0=是）'] == 1])
print(noenlarge_in_give_up)
# 概率计算
p_noenlarge_in_stick_to = noenlarge_in_stick_to / graduate_total_stcik_to
print(p_noenlarge_in_stick_to)
p_noenlarge_in_give_up = noenlarge_in_give_up / graduate_total_give_up
print(p_noenlarge_in_give_up)

# 统计和计算分别在坚持和放弃考研的条件下 新冠疫情使您的备考过程是否更加困难的人数和概率
# 人数统计
is_difficult_in_stick_to = len(graduate_data_stick_to[graduate_data_stick_to['新冠疫情使您的备考过程更加困难（0=是）'] == 0])
print(is_difficult_in_stick_to)
is_difficult_in_give_up = len(graduate_data_give_up[graduate_data_give_up['新冠疫情使您的备考过程更加困难（0=是）'] == 0])
print(is_difficult_in_give_up)
# 概率计算
p_is_difficult_in_stick_to = is_difficult_in_stick_to / graduate_total_stcik_to
print(p_is_difficult_in_stick_to)
p_is_difficult_in_give_up = is_difficult_in_give_up / graduate_total_give_up
print(p_is_difficult_in_give_up)

# 人数统计
no_difficult_in_stick_to = len(graduate_data_stick_to[graduate_data_stick_to['新冠疫情使您的备考过程更加困难（0=是）'] == 1])
print(no_difficult_in_stick_to)
no_difficult_in_give_up = len(graduate_data_give_up[graduate_data_give_up['新冠疫情使您的备考过程更加困难（0=是）'] == 1])
print(no_difficult_in_give_up)
# 概率计算
p_no_difficult_in_stick_to = no_difficult_in_stick_to / graduate_total_stcik_to
print(p_no_difficult_in_stick_to)
p_no_difficult_in_give_up = no_difficult_in_give_up / graduate_total_give_up
print(p_no_difficult_in_give_up)

# 建立分类器概率字典
bayias_claasifier_stick_to = {'坚持考研':p_graduate_total_stcik_to, '男生|坚持考研':p_boy_in_stick_to, '女生|坚持考研':p_girl_in_stick_to, '焦虑|坚持考研':p_is_anxious_in_stick_to,
                              '不焦虑|坚持考研':p_no_anxious_in_stick_to, '积极|坚持考研':p_postive_in_stick_to, '消极|坚持考研':p_negtive_in_stick_to, '二者之间|坚持考研':p_between_in_stick_to,
                              '有损失|坚持考研':p_is_loss_in_stick_to, '没损失|坚持考研':p_no_loss_in_stick_to, '扩大招生|坚持考研':p_enlarge_in_stick_to, '没扩大招生|坚持考研':p_noenlarge_in_stick_to,
                              '备考更加困难|坚持考研':p_is_difficult_in_stick_to, '备考没有更加困难|坚持考研':p_no_difficult_in_stick_to}

bayias_claasifier_give_up = {'放弃考研':p_graduate_total_give_up, '男生|放弃考研':p_boy_in_give_up, '女生|放弃考研':p_girl_in_give_up, '焦虑|放弃考研':p_is_anxious_in_give_up,
                              '不焦虑|放弃考研':p_no_anxious_in_give_up, '积极|放弃考研':p_positive_in_give_up, '消极|放弃考研':p_negtive_in_give_up, '二者之间|放弃考研':p_between_in_give_up,
                              '有损失|放弃考研':p_is_loss_in_give_up, '没损失|放弃考研':p_no_loss_in_give_up, '扩大招生|放弃考研':p_enlarge_in_give_up, '没扩大招生|放弃考研':p_noenlarge_in_give_up,
                              '备考更加困难|放弃考研':p_is_difficult_in_give_up, '备考没有更加困难|放弃考研':p_no_difficult_in_give_up}

def calculate_p_in_stick_and_give(gender,motion,antitude,fanical,recruit,prepare_exam):
    gender_stick = gender + '|坚持考研'
    motion_stick = motion + '|坚持考研'
    antitude_stick = antitude + '|坚持考研'
    fanical_stick = fanical + '|坚持考研'
    recruit_stick = recruit + '|坚持考研'
    prepare_exam_stick = prepare_exam + '|坚持考研'

    gender_giveup = gender + '|放弃考研'
    motion_giveup = motion + '|放弃考研'
    antitude_giveup = antitude + '|放弃考研'
    fanical_giveup = fanical + '|放弃考研'
    recruit_giveup = recruit + '|放弃考研'
    prepare_exam_giveup = prepare_exam + '|放弃考研'

    # 计算预测坚持考研的概率
    p_stick = bayias_claasifier_stick_to['坚持考研'] * bayias_claasifier_stick_to[gender_stick] * bayias_claasifier_stick_to[motion_stick] \
              * bayias_claasifier_stick_to[antitude_stick] * bayias_claasifier_stick_to[fanical_stick] * bayias_claasifier_stick_to[recruit_stick] \
              * bayias_claasifier_stick_to[prepare_exam_stick]

    p_giveup = bayias_claasifier_give_up['放弃考研'] * bayias_claasifier_give_up[gender_giveup] * bayias_claasifier_give_up[motion_giveup] \
              * bayias_claasifier_give_up[antitude_giveup] * bayias_claasifier_give_up[fanical_giveup] * bayias_claasifier_give_up[recruit_giveup] \
              * bayias_claasifier_give_up[prepare_exam_giveup]

    return p_stick, p_giveup

if __name__ == '__main__':
    p_stick, p_giveup = calculate_p_in_stick_and_give('男生','不焦虑','积极','没损失','扩大招生','备考没有更加困难')
    print(f'在该特征属性条件下坚持考研的概率为{p_stick}')
    print(f'在该特征属性条件下放弃考研的概率为{p_giveup}')
