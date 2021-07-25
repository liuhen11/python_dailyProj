class Person(object):
    #会员信息类
    def __init__(self, id, name, passcode, gender, birth, level, score):
        self.id = id                         #编号
        self.name = name                   	#姓名
        self.passcode = passcode             #密码
        self.gender = gender						#性别
        self.birth = birth 						#生日
        self.level = level                 #等级
        self.score = score                #积分
