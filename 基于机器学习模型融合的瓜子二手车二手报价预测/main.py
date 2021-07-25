
from PyQt5.QtWidgets import QApplication, QMessageBox
# from PyQt5.QtUiTools import QUiLoader
from PyQt5 import uic
from PyQt5.Qt import *
from PyQt5.QtWidgets import qApp,QFileDialog
#from PyQt5.QtCore import QUrl, pyqtSignal, QEventLoop
import matplotlib.ticker as ticker
from PyQt5 import QtCore, QtGui
from PyQt5.QtWebEngineWidgets import *
from PyQt5 import QtWidgets
import sys
import pandas as pd
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import requests
import matplotlib
matplotlib.use("Qt5Agg")  # 声明使用QT5
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib as mpl
mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['font.serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False # 解决保存图像是负号'-'显示为方块的问题,或者转换负号为字符串

## 基础工具
import numpy as np
import pandas as pd
import warnings
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.special import jn
from IPython.display import display, clear_output
import time

warnings.filterwarnings('ignore')
## 模型预测的
from sklearn import linear_model
from sklearn import preprocessing
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

## 数据降维处理的
from sklearn.decomposition import PCA, FastICA, FactorAnalysis, SparsePCA

import lightgbm as lgb
import xgboost as xgb

## 参数搜索和评价
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

from lib.share import SI
root=""
fileNum = 0
myrow=0

#自定义函数SaveExcel用于保存数据到Excel
def SaveExcel(df,isChecked,x):
    # 将提取后的数据保存到Excel
    if (isChecked):
        df.to_csv(x)
    else:
        df.to_csv(x)

class EmittingStr(QtCore.QObject):
    textWritten = QtCore.pyqtSignal(str) #定义一个发送str的信号
    def write(self, text):
      self.textWritten.emit(str(text))

class Main_window():

    def __init__(self):
        # 从文件中加载UI定义

        # 从 UI 定义中动态 创建一个相应的窗口对象
        # 注意：里面的控件对象也成为窗口对象的属性了
        # 比如 self.ui.button , self.ui.textEdit
        self.ui = uic.loadUi('datacsv.ui')
        self.ui.button7.triggered.connect(self.closeEvent)
        self.ui.button_importCsv.triggered.connect(self.click_import_cb)
        self.ui.button_dataProcess.triggered.connect(self.dataProcess)
        self.ui.button_viewAnalysis.triggered.connect(self.viewAnalysis)
        self.ui.button_bigdataMakemodel.triggered.connect(self.bigdataMakemodel)
        self.ui.button_viewPredict.triggered.connect(self.predictResult_)
        self.ui.list1.clicked.connect(self.clicked)
        # 单击"浏览"按钮，选择文件存储路径
        self.ui.viewButton.clicked.connect(self.save_clicked)
        # # 设置Dataframe对象显示所有列
        pd.set_option('display.max_columns', None)
        # # 设置Dataframe对象列宽为200，默认为50
        pd.set_option('max_colwidth', 200)
        #self.ui.btn_login.clicked.connect(self.onSignIn)
        # self.ui.edt_password.returnPressed.connect(self.onSignIn)

    def closeEvent(self, event):
        sys.exit(0)

    def save_clicked(self):
        curPath=QDir.currentPath()  #获取系统当前目录
        dlgTitle = "保存文件"
        filt="所有文件(*.*);;文本文件(*.csv);;图片文件(*.jpg*.gif*.png)"
        #temproot = QFileDialog.getExistingDirectory(self, "选择文件夹", "/")
        filename,filtUsed=QFileDialog.getSaveFileName(self.ui,dlgTitle,curPath,filt)
        if filename == "":
            print("\n取消选择")
            return
        self.ui.text1.setText(filename)
        if self.ui.rButton1.isChecked():
            SaveExcel(self.df1, self.ui.rButton1.isChecked(), filename)
        #print(type(d))
        # self.ui.text1.appendPlainText(filename)
        # self.ui.text1.appendPlainText("\n"+filtUsed)

    # 导入csv文件回调函数
    def click_import_cb(self):
        # 文件夹路径
        global root
        root = QFileDialog.getExistingDirectory(self.ui, "选择文件夹", "/")
        if root == "":
            print("\n取消选择")
            return
        mylist = []
        # 遍历文件夹文件
        for dirpath, dirnames, filenames in os.walk(root):
            for filepath in filenames:
                # mylist.append(os.path.join(dirpath, filepath))
                mylist.append(os.path.join(filepath))
        # 实例化列表模型，添加数据列表
        self.model = QtCore.QStringListModel()
        # 添加列表数据
        self.model.setStringList(mylist)
        self.ui.list1.setModel(self.model)
        self.ui.list1 = mylist

    def clicked(self, qModelIndex):
        global root
        global myrow
        myrow=qModelIndex.row()
        # 获取当前选中行的数据
        self.a = root + '/' + str(self.ui.list1[qModelIndex.row()])
        if '.csv' in self.a:
            df = pd.DataFrame(pd.read_csv(self.a))
            self.ui.textEdit.setText(str(df))
        else:
            QMessageBox.critical(self.ui,'错误','请选择文件名带有column的csv文件！')

    def dataProcess(self):
        choice = QMessageBox.question(
            self.ui,
            '前情提醒',
            '是否选择了csv文件？')

        if choice == QMessageBox.Yes:
            print('你选择了yes')
            global root
            global myrow
            carData = pd.read_csv(self.a)
            carData2 = carData.copy()
            carData1 = carData2.drop('Unnamed: 0', axis=1)
            self.df1 = carData1[carData1['变速箱'].isin(['自动', '手动'])]
            self.df1['变速箱'].replace({'自动': 0, '手动': 1}, inplace=True)
            self.df1['里程（万公里）'] = self.df1['里程（万公里）'].str.split('万公里', expand=True)[0].apply(pd.to_numeric,
                                                                                             errors='ignore')
            self.df1['车牌注册日期'] = self.df1['车牌注册日期'].str.split('年', expand=True)[0].apply(pd.to_numeric, errors='ignore')
            self.df1['车龄'] = 2021 - self.df1['车牌注册日期']
            print(self.df1)

            self.ui.textEdit.setText(str(self.df1))
            SaveExcel(self.df1, self.ui.rButton2.isChecked(),'after_process_car.csv')
        else:
            print('你选择了no')
            QMessageBox.critical(self.ui, '错误', '请先选择需要处理的csv文件！')
        # global root
        # global myrow
        # carData = pd.read_csv(self.a)
        # carData2 = carData.copy()
        # carData1 = carData2.drop('Unnamed: 0', axis=1)
        # self.df1 = carData1[carData1['变速箱'].isin(['自动', '手动'])]
        # self.df1['变速箱'].replace({'自动': 0, '手动': 1}, inplace=True)
        # self.df1['里程（万公里）'] = self.df1['里程（万公里）'].str.split('万公里', expand=True)[0].apply(pd.to_numeric, errors='ignore')
        # self.df1['车牌注册日期'] = self.df1['车牌注册日期'].str.split('年', expand=True)[0].apply(pd.to_numeric, errors='ignore')
        # self.df1['车龄'] = 2021 - self.df1['车牌注册日期']
        # print(self.df1)
        #
        # self.ui.textEdit.setText(str(self.df1))
        # else:
        #     QMessageBox.critical(self.ui, '错误', '请先选择需要处理的csv文件！')

    def viewAnalysis(self):

        SI.mainWin = Win_Main()
        SI.mainWin.ui.show()

    def bigdataMakemodel(self):
        SI.mlpreWin = CarPredictmodel()
        SI.mlpreWin.ui.show()

    def predictResult_(self):
        SI.pre = MypredictResult()
        SI.pre.ui.show()
        #
        # self.ui.edt_password.setText('')
        # self.ui.hide()

class CarPredictmodel :
    def __init__(self):
        self.ui = uic.loadUi('ml_predict_model.ui')
        self.ui.pushButton_Eda.clicked.connect(self.exportEdareport)
        self.ui.pushButton_train.clicked.connect(self.startTrain)

        # 下面将输出重定向到textBrowser中
        sys.stdout = EmittingStr(textWritten=self.outputWritten)
        sys.stderr = EmittingStr(textWritten=self.outputWritten)

    def outputWritten(self, text):
        cursor = self.ui.textBrowser_train.textCursor()
        cursor.movePosition(QtGui.QTextCursor.End)
        cursor.insertText(text)
        self.ui.textBrowser_train.setTextCursor(cursor)
        self.ui.textBrowser_train.ensureCursorVisible()

    # 创建导出数据探索性分析报告函数
    def exportEdareport(self):
        import warnings
        warnings.filterwarnings('ignore')

        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        import seaborn as sns
        import missingno as msno

        choice = QMessageBox.question(
            self.ui,
            '前情提醒',
            '是否已经预处理了数据？')
        if choice == QMessageBox.Yes:
            msno.matrix(SI.loginWin.df1)
            msno.bar(SI.loginWin.df1)
            SI.loginWin.df1['车型_品牌'].value_counts()
            SI.loginWin.df1['车主报价（万）'].value_counts()
            SI.loginWin.df1 = SI.loginWin.df1[SI.loginWin.df1['车主报价（万）'] <= 40]
            import scipy.stats as st
            y = SI.loginWin.df1['车主报价（万）']
            plt.figure(1);
            plt.title('Johnson SU')
            sns.distplot(y, kde=False, fit=st.johnsonsu)
            plt.figure(2);
            plt.title('Normal')
            sns.distplot(y, kde=False, fit=st.norm)
            plt.figure(3);
            plt.title('Log Normal')
            sns.distplot(y, kde=False, fit=st.lognorm)
            ## 2) 查看skewness and kurtosis
            sns.distplot(SI.loginWin.df1['车主报价（万）']);
            print("Skewness: %f" % SI.loginWin.df1['车主报价（万）'].skew())
            print("Kurtosis: %f" % SI.loginWin.df1['车主报价（万）'].kurt())
            sns.distplot(SI.loginWin.df1.skew(), color='blue', axlabel='Skewness')
            sns.distplot(SI.loginWin.df1.kurt(), color='orange', axlabel='Kurtness')
            ## 3) 查看预测值的具体频数
            plt.hist(SI.loginWin.df1['车主报价（万）'], orientation='vertical', histtype='bar', color='red')
            plt.show()
            # log变换 z之后的分布较均匀，可以进行log变换进行预测，这也是预测问题常用的trick
            plt.hist(np.log(SI.loginWin.df1['车主报价（万）']), orientation='vertical', histtype='bar', color='red')
            plt.show()
            # 分离预测值
            from sklearn.utils import shuffle
            SI.loginWin.df1 = shuffle(SI.loginWin.df1)
            self.train_data = SI.loginWin.df1.iloc[0:25000]
            self.test_data = SI.loginWin.df1.iloc[25000:]
            y_train = self.train_data['车主报价（万）']
            numeric_features = ['车龄', '里程（万公里）', '新车指导价（万）']
            categorical_features = ['车型_品牌', '排量', '变速箱']
            # 特征nunique分布
            for cat_fea in categorical_features:
                print(cat_fea + "的特征分布如下：")
                print("{}特征有个{}不同的值".format(cat_fea, self.train_data[cat_fea].nunique()))
                print(self.train_data[cat_fea].value_counts())
            # 特征nunique分布
            for cat_fea in categorical_features:
                print(cat_fea + "的特征分布如下：")
                print("{}特征有个{}不同的值".format(cat_fea, self.test_data[cat_fea].nunique()))
                print(self.test_data[cat_fea].value_counts())
            numeric_features.append('车主报价（万）')
            ## 1) 相关性分析
            price_numeric = self.train_data[numeric_features]
            correlation = price_numeric.corr()
            print(correlation['车主报价（万）'])
            f, ax = plt.subplots(figsize=(7, 7))
            plt.title('Correlation of Numeric Features with Price', y=1, size=16)
            sns.heatmap(correlation, square=True, vmax=0.8)
            del price_numeric['车主报价（万）']

            ## 2) 查看几个特征的偏度和峰值
            for col in numeric_features:
                print('{:15}'.format(col),
                      'Skewness: {:05.2f}'.format(self.train_data[col].skew()),
                      '   ',
                      'Kurtosis: {:06.2f}'.format(self.train_data[col].kurt())
                      )

            ## 3) 每个数字特征得分布可视化
            f = pd.melt(self.train_data, value_vars=numeric_features)
            g = sns.FacetGrid(f, col="variable", col_wrap=2, sharex=False, sharey=False)
            g = g.map(sns.distplot, "value")

            ## 4) 数字特征相互之间的关系可视化
            columns = ['车主报价（万）', '车龄', '里程（万公里）', '新车指导价（万）']
            sns.pairplot(self.train_data[columns], size=2, kind='scatter', diag_kind='kde')
            plt.show()

            ## 5) 多变量互相回归关系可视化
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, figsize=(24, 20))
            v_12_scatter_plot = pd.concat([y_train, self.train_data['新车指导价（万）']], axis=1)
            sns.regplot(x='新车指导价（万）', y='车主报价（万）', data=v_12_scatter_plot, scatter=True, fit_reg=True, ax=ax1)
            v_8_scatter_plot = pd.concat([y_train, self.train_data['里程（万公里）']], axis=1)
            sns.regplot(x='里程（万公里）', y='车主报价（万）', data=v_8_scatter_plot, scatter=True, fit_reg=True, ax=ax2)
            v_0_scatter_plot = pd.concat([y_train, self.train_data['车龄']], axis=1)
            sns.regplot(x='车龄', y='车主报价（万）', data=v_0_scatter_plot, scatter=True, fit_reg=True, ax=ax3)

            ## 1) unique分布
            for fea in categorical_features:
                print(self.train_data[fea].nunique())

            ## 2) 类别特征箱形图可视化
            categorical_features = ['排量', '变速箱']
            for c in categorical_features:
                self.train_data[c] = self.train_data[c].astype('category')
                if self.train_data[c].isnull().any():
                    self.train_data[c] = self.train_data[c].cat.add_categories(['MISSING'])
                    self.train_data[c] = self.train_data[c].fillna('MISSING')
            def boxplot(x, y, **kwargs):
                sns.boxplot(x=x, y=y)
                x = plt.xticks(rotation=90)
            f = pd.melt(self.train_data, id_vars=['车主报价（万）'], value_vars=categorical_features)
            g = sns.FacetGrid(f, col="variable", col_wrap=2, sharex=False, sharey=False, size=5)
            g = g.map(boxplot, "value", "车主报价（万）")

            ## 3) 类别特征的小提琴图可视化
            catg_list = categorical_features
            target = '车主报价（万）'
            for catg in catg_list:
                sns.violinplot(x=catg, y=target, data=self.train_data)
                plt.show()

            ## 4) 类别特征的柱形图可视化
            def bar_plot(x, y, **kwargs):
                sns.barplot(x=x, y=y)
                x = plt.xticks(rotation=90)

            f = pd.melt(self.train_data, id_vars=['车主报价（万）'], value_vars=categorical_features)
            g = sns.FacetGrid(f, col="variable", col_wrap=2, sharex=False, sharey=False, size=5)
            g = g.map(bar_plot, "value", "车主报价（万）")

            ##  5) 类别特征的每个类别频数可视化(count_plot)
            def count_plot(x, **kwargs):
                sns.countplot(x=x)
                x = plt.xticks(rotation=90)

            f = pd.melt(self.train_data, value_vars=categorical_features)
            g = sns.FacetGrid(f, col="variable", col_wrap=2, sharex=False, sharey=False, size=5)
            g = g.map(count_plot, "value")
            plt.show()

            import pandas_profiling
            pfr = pandas_profiling.ProfileReport(self.train_data)
            pfr.to_file("./二手车数据分析EDA报告.html")
            url = os.getcwd()+'/二手车数据分析EDA报告.html'
            self.ui.webEngineView.load(QtCore.QUrl.fromLocalFile(url))
            #self.ui.webEngineView.load(QUrl.fromLocalFile("./二手车数据分析EDA报告.html"))
            #self.ui.setCentralWidget(self.ui.webEngineView)
        else:
            print('你选择了no')
            QMessageBox.critical(self.ui, '错误', '请先对数据进行处理！')
    ## 定义了一个统计函数，方便后续信息统计
    def Sta_inf(self,data):
        print('_min', np.min(data))
        print('_max:', np.max(data))
        print('_mean', np.mean(data))
        print('_ptp', np.ptp(data))
        print('_std', np.std(data))
        print('_var', np.var(data))

    def getXtain_Ytrain_Xtest(self):

        print('train data shape:', self.train_data.shape)
        print('test data shape:', self.test_data.shape)

        numerical_cols_ = self.train_data.select_dtypes(exclude='object').columns
        print(numerical_cols_)
        categorical_cols_ = self.train_data.select_dtypes(include='object').columns
        print(categorical_cols_)
        # 选择特征列
        feature_cols = [col for col in numerical_cols_ if col not in ['车牌注册日期', '车主报价（万）', '排量', '变速箱']]
        feature_cols = [col for col in feature_cols if 'Type' not in col]
        ## 提前特征列，标签列构造训练样本和测试样本
        self.X_data = self.train_data[feature_cols]
        self.Y_data = self.train_data['车主报价（万）']

        self.X_test = self.test_data[feature_cols]

        print('X train shape:', self.X_data.shape)
        print('X test shape:', self.X_test.shape)


        print('Sta of label:')
        self.Sta_inf(self.Y_data)

        ## 绘制标签的统计图，查看标签分布
        plt.hist(self.Y_data)
        plt.show()
        plt.close()

        x_train, x_val, y_train,y_val = train_test_split(self.X_data, self.Y_data, test_size=0.3, random_state=42)
        return x_train, x_val, y_train,y_val

    def build_model_xgb(self,x_train, y_train, n_estimators, learning_rate, gamma):
        model = xgb.XGBRegressor(n_estimators=n_estimators, learning_rate=learning_rate, gamma=gamma, subsample=0.8, \
                                 colsample_bytree=0.9, max_depth=7)  # , objective ='reg:squarederror'
        model.fit(x_train, y_train)
        return model

    def build_model_lgb(self,x_train, y_train, num_leaves, n_estimators):
        estimator = lgb.LGBMRegressor(num_leaves=num_leaves, n_estimators=n_estimators)
        param_grid = {
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
        }
        gbm = GridSearchCV(estimator, param_grid)
        gbm.fit(x_train, y_train)
        return gbm

    def startTrain(self):
        # 调用生成验证集函数，将训练数据集分割为训练集和验证集
        self.x_train, self.x_val, self.y_train, self.y_val = self.getXtain_Ytrain_Xtest()
        if self.ui.comboBox.currentText() == 'XGBoost预测模型':
            xgb_lr = float(self.ui.lineEdit_Xgblr.text())
            xgb_gamma = float(self.ui.lineEdit_Xgbgamma.text())
            xgb_est = int(self.ui.lineEdit_Xgbest.text())

            print('Train xgb...')
            model_xgb = self.build_model_xgb(self.x_train, self.y_train, xgb_est, xgb_lr, xgb_gamma)
            self.val_xgb = model_xgb.predict(self.x_val)
            self.MAE_xgb = mean_absolute_error(self.y_val, self.val_xgb)
            print('MAE of val with xgb:', self.MAE_xgb)

            print('Predict xgb...')
            model_xgb_pre = self.build_model_xgb(self.X_data, self.Y_data, xgb_est, xgb_lr, xgb_gamma)
            self.subA_xgb = model_xgb_pre.predict(self.X_test)
            print('Sta of Predict xgb:')
            self.Sta_inf(self.subA_xgb)

            subA = pd.DataFrame()
            subA['车型_品牌'] = self.test_data.车型_品牌
            subA['车主报价预测'] = self.subA_xgb
            subA.to_csv('./subA_xgb_Weighted.csv', index=False)
            QMessageBox.information(self.ui, '操作成功', '模型训练结束')
        elif self.ui.comboBox.currentText() == 'Light GBM预测模型':
            lgb_numleav = int(self.ui.lineEdit_Lgbnum.text())
            lgb_est = int(self.ui.lineEdit_Lgbest.text())

            print('Train lgb...')
            model_lgb = self.build_model_lgb(self.x_train, self.y_train, lgb_numleav, lgb_est)
            self.val_lgb = model_lgb.predict(self.x_val)
            self.MAE_lgb = mean_absolute_error(self.y_val, self.val_lgb)
            print('MAE of val with lgb:', self.MAE_lgb)

            print('Predict lgb...')
            model_lgb_pre = self.build_model_lgb(self.X_data, self.Y_data, lgb_numleav, lgb_est)
            self.subA_lgb = model_lgb_pre.predict(self.X_test)
            print('Sta of Predict lgb:')
            self.Sta_inf(self.subA_lgb)

            subB = pd.DataFrame()
            subB['车型_品牌'] = self.test_data.车型_品牌
            subB['车主报价预测'] = self.subA_lgb
            subB.to_csv('./subA_lgb_Weighted.csv', index=False)
            QMessageBox.information(self.ui, '操作成功', '模型训练结束')
        else:
            # 进行模型融合
            ## 这里我们采取了简单的加权融合的方式
            val_Weighted = (1 - self.MAE_lgb / (self.MAE_xgb + self.MAE_lgb)) * self.val_lgb + (1 - self.MAE_xgb / (self.MAE_xgb + self.MAE_lgb)) * self.val_xgb
            val_Weighted[val_Weighted < 0] = 10  # 由于我们发现预测的最小值有负数，而真实情况下，price为负是不存在的，由此我们进行对应的后修正
            print('MAE of val with Weighted ensemble:', mean_absolute_error(self.y_val, val_Weighted))

            sub_Weighted = (1 - self.MAE_lgb / (self.MAE_xgb + self.MAE_lgb)) * self.subA_lgb + (
                        1 - self.MAE_xgb / (self.MAE_xgb + self.MAE_lgb)) * self.subA_xgb

            ## 查看预测值的统计进行
            plt.hist(self.Y_data)
            plt.show()
            plt.close()

            sub = pd.DataFrame()
            sub['车型_品牌'] = self.test_data.车型_品牌
            sub['车主报价预测'] = sub_Weighted
            sub.to_csv('./sub_Weighted.csv', index=False)
            self.test_data.to_csv('./testdata.csv',index=False)
            QMessageBox.information(self.ui, '操作成功', '模型训练结束')

class Win_Main :
    def __init__(self):
        self.ui = uic.loadUi('viewresult.ui')
        self.ui.pushButton_viewConfirm.clicked.connect(self.viewConfirmcb)
        # 单击"浏览"按钮，选择文件存储路径
        self.ui.pushButton_2.clicked.connect(self.save_clicked_)


    def save_clicked_(self):
        curPath_=QDir.currentPath()  #获取系统当前目录
        dlgTitle_ = "保存文件"
        filt_="所有文件(*.*);;文本文件(*.csv);;图片文件(*.jpg)"
        #temproot = QFileDialog.getExistingDirectory(self, "选择文件夹", "/")
        filename_,filtUsed_=QFileDialog.getSaveFileName(self.ui,dlgTitle_,curPath_,filt_)
        if filename_ == "":
            print("\n取消选择")
            return
        self.ui.textEdit.setText(filename_)
        x = filename_.split('.jpg')
        if self.ui.comboBox.currentText() != '总体分析':
            fig1s = self.F1.axes.get_figure()
            fig1s.savefig(x[0]+'1.jpg')
            fig2s = self.F2.axes.get_figure()
            fig2s.savefig(x[0] + '2.jpg')
            fig3s = self.F3.axes.get_figure()
            fig3s.savefig(x[0] + '3.jpg')
            fig4s = self.F4.axes.get_figure()
            fig4s.savefig(x[0] + '4.jpg')
            fig5s = self.F5.axes.get_figure()
            fig5s.savefig(x[0] + '5.jpg')
            fig6s = self.F6.axes.get_figure()
            fig6s.savefig(x[0] + '6.jpg')
        else:
            figs = self.F.axes.get_figure()
            figs.savefig(x[0] + '总体.jpg')

    def viewConfirmcb(self):
        #c = self.ui.comboBox.currentText()
        if self.ui.comboBox.currentText() != '总体分析':
            self.F1 = MyFigure()
            self.F2 = MyFigure()
            self.F3 = MyFigure()
            self.F4 = MyFigure()
            self.F5 = MyFigure()
            self.F6 = MyFigure()
            # self.F.plotsin()
            self.carAge_number()
            # # 第六步：在GUI的groupBox中创建一个布局，用于添加MyFigure类的实例（即图形）后其他部件。
            #   # 继承容器groupBox
            self.ui.gridLayout.addWidget(self.F1, 0, 0)
            self.kiloMeter_discount()
            self.ui.gridLayout.addWidget(self.F2, 0, 1)
            self.carDate_discount()
            self.ui.gridLayout.addWidget(self.F3, 0, 2)
            self.carSpeedT_number()
            self.ui.gridLayout.addWidget(self.F4, 1, 0)
            self.secondPrice_number()
            self.ui.gridLayout.addWidget(self.F5, 1, 1)
            self.newGuidePrice_number()
            self.ui.gridLayout.addWidget(self.F6, 1, 2)
        else:
            self.F = MyFigure(width=6, height=4, dpi=200)
            self.conclusionAnaly()
            self.ui.gridLayout.addWidget(self.F, 0, 0,2,3)

    def carAge_number(self):
        self.dazhong_data = SI.loginWin.df1.loc[SI.loginWin.df1['车型_品牌'].str.contains(self.ui.comboBox.currentText())]
        age_value = self.dazhong_data['车型_品牌'].groupby(self.dazhong_data['车龄']).count().tolist()
        age_label = self.dazhong_data['车龄'].unique()
        age_label = np.sort(age_label).tolist()
        self.F1.axes.bar(age_label, age_value)
        self.F1.axes.set_title('不同车龄段二手车数量')
        self.F1.axes.set_xlabel('车龄')
        self.F1.axes.set_ylabel('二手车数量')
        self.F1.axes.tick_params(direction='in')
        self.F1.axes.tick_params(direction='in')
        self.F1.fig.tight_layout(pad=1.8)
        # figs = self.F1.axes.get_figure()
        # figs.savefig("output.png")

    def kiloMeter_discount(self):
        discount = (self.dazhong_data.loc[:, '新车指导价（万）'] - self.dazhong_data.loc[:, '车主报价（万）']) * 100 / self.dazhong_data.loc[:,                                                                                    '新车指导价（万）']
        self.dazhong_data['折扣率'] = discount
        self.F2.axes.scatter(self.dazhong_data['里程（万公里）'], discount)
        self.F2.axes.set_xlabel('里程（万公里）')
        self.F2.axes.set_ylabel('折扣率(%)')
        #plt.title('折扣率与公里数散点图')
        self.F2.axes.set_title('折扣率与里程数散点图')
        self.F2.axes.tick_params(direction='in')
        self.F2.axes.tick_params(direction='in')
        self.F2.fig.tight_layout(pad=1.8)
        SI.loginWin.ui.textEdit.setText(str(self.dazhong_data))

    def carDate_discount(self):
        self.F3.axes.scatter(self.dazhong_data['车牌注册日期'], self.dazhong_data['折扣率'])
        self.F3.axes.set_xlabel('车牌注册日期')
        self.F3.axes.set_ylabel('折扣率(%)')
        # plt.title('折扣率与公里数散点图')
        self.F3.axes.set_title('折扣率与车牌注册日期散点图')
        self.F3.axes.tick_params(direction='in')
        self.F3.axes.tick_params(direction='in')
        self.F3.fig.tight_layout(pad=1.8)

    def carSpeedT_number(self):
        gear_data_value = self.dazhong_data['车型_品牌'].groupby(self.dazhong_data['变速箱']).count().tolist()
        if len(gear_data_value) == 1:
            gear_data_value.append(0)
        gear_data_label = ['自动变速', '手动变速', '带涡轮增压', '不带涡轮增压']
        pailiang_data_value = [self.dazhong_data.loc[self.dazhong_data['排量'].str.contains('T')].shape[0],
                               self.dazhong_data.loc[self.dazhong_data['排量'].str.contains('L')].shape[0]]
        pailiang_gear = [gear_data_value[0], gear_data_value[1], pailiang_data_value[0], pailiang_data_value[1]]
        self.F4.axes.bar(gear_data_label, pailiang_gear)
        self.F4.axes.set_xlabel('汽车减速箱和排量情况')
        #self.F4.axes.set_xticklabels(rotation=15)
        self.F4.axes.set_ylabel('二手车数量')
        # plt.title('折扣率与公里数散点图')
        self.F4.axes.set_title('汽车不同性能与二手车数量')
        self.F4.axes.tick_params(direction='in')
        self.F4.axes.tick_params(direction='in')
        self.F4.fig.autofmt_xdate(rotation=15)
        self.F4.fig.tight_layout(pad=2.0)

    def secondPrice_number(self):
        self.dazhong_data['车主报价区间'] = pd.cut(x=self.dazhong_data["车主报价（万）"], bins=6)
        baojia_value = self.dazhong_data['车型_品牌'].groupby(self.dazhong_data['车主报价区间']).count()
        self.F5.axes.plot(baojia_value)
        positions = [0, 1, 2, 3, 4, 5]
        self.F5.axes.xaxis.set_major_locator(ticker.FixedLocator(positions))
        self.F5.axes.xaxis.set_major_formatter(ticker.FixedFormatter(baojia_value.index))
        self.F5.axes.grid(which='major', axis='both', linestyle='-')
        self.F5.axes.set_xlabel('车主报价区间分布')
       # self.F5.axes.set_xticklabels(baojia_value.index)
        self.F5.axes.set_ylabel('二手车数量')
        # plt.title('折扣率与公里数散点图')
        self.F5.axes.set_title('二手报价价格区间与二手车数量')
        self.F5.axes.tick_params(direction='in')
        self.F5.axes.tick_params(direction='in')
        self.F5.fig.autofmt_xdate(rotation=30)
        self.F5.fig.tight_layout(pad=2.0)
    def newGuidePrice_number(self):
        self.dazhong_data['新车指导价区间'] = pd.cut(x=self.dazhong_data["新车指导价（万）"], bins=6)
        baojia_value = self.dazhong_data['车型_品牌'].groupby(self.dazhong_data['新车指导价区间']).count()
        self.F6.axes.plot(baojia_value)
        positions = [0, 1, 2, 3, 4, 5]
        self.F6.axes.xaxis.set_major_locator(ticker.FixedLocator(positions))
        self.F6.axes.xaxis.set_major_formatter(ticker.FixedFormatter(baojia_value.index))
        self.F6.axes.grid(which='major', axis='both', linestyle='-')
        self.F6.axes.set_xlabel('新车指导价区间分布')
        # self.F5.axes.set_xticklabels(baojia_value.index)
        self.F6.axes.set_ylabel('二手车数量')
        # plt.title('折扣率与公里数散点图')
        self.F6.axes.set_title('新车指导价区间与二手车数量')
        self.F6.axes.tick_params(direction='in')
        self.F6.axes.tick_params(direction='in')
        self.F6.fig.autofmt_xdate(rotation=30)
        self.F6.fig.tight_layout(pad=1.8)

    def conclusionAnaly(self):
        # 总体分析
        listxx = ['大众', '奔驰', '宝马', '奥迪', '日产', '丰田', '本田', '马自达', '哈弗', '东风', '别克', '福特', '现代', '雪佛兰', '吉利', '斯巴鲁',
                  '雷克萨斯']
        che_count = {}
        for i in range(len(listxx)):
            che_count[listxx[i]] = SI.loginWin.df1.loc[SI.loginWin.df1['车型_品牌'].str.contains(listxx[i])].shape[0]
        self.F.axes.bar(che_count.keys(), che_count.values())
        self.F.axes.set_xlabel('车型品牌')
        # self.F5.axes.set_xticklabels(baojia_value.index)
        self.F.axes.set_ylabel('二手车数量')
        # plt.title('折扣率与公里数散点图')
        self.F.axes.set_title('不同品牌的二手车数量')
        self.F.axes.tick_params(direction='in')
        self.F.axes.tick_params(direction='in')
        self.F.fig.autofmt_xdate(rotation=45)
        self.F.fig.tight_layout(pad=1.8)

# 创建预测结果可视化类
class MypredictResult:
    def __init__(self):
        self.ui = uic.loadUi('predict_resu_view.ui')
        # 单击"保存"按钮，选择文件存储路径
        if os.path.exists('./subA_xgb_Weighted.csv') and os.path.exists('./subA_lgb_Weighted.csv') and os.path.exists('./sub_Weighted.csv')and os.path.exists('./testdata.csv'):
            self.xgb = pd.read_csv('./subA_xgb_Weighted.csv')
            self.data_xgb = self.xgb.sample(100)
            self.lgb = pd.read_csv('./subA_lgb_Weighted.csv')
            self.lgb_xgb = pd.read_csv('./sub_Weighted.csv')
            self.testdata = pd.read_csv('./testdata.csv')
            self.F1 = MyFigure()
            self.F2 = MyFigure()
            self.F3 = MyFigure()
            self.F4 = MyFigure()

                #   # 继承容器groupBox

            self.xgbplot()
            self.ui.gridLayout.addWidget(self.F1, 0, 0)
            self.lgbtyplot()
            self.ui.gridLayout.addWidget(self.F2, 0, 1)
            self.lgbxgb()
            self.ui.gridLayout.addWidget(self.F3, 1, 0)
            self.piethree()
            self.ui.gridLayout.addWidget(self.F4, 1, 1)
        else:
            QMessageBox.information(self.ui, '操作失败', '本地数据库无预测结果数据')
        self.ui.pushButton.clicked.connect(self.pushButtonsaveclick)
        self.ui.pushButton_2.clicked.connect(self.pushButton_save_click_)

    def pushButtonsaveclick(self):
        self.temproot_ = QFileDialog.getExistingDirectory(self.ui, "选择文件夹", "/")
        # if self.temproot == "":
        #     print("\n取消选择")
        #     return
        self.ui.lineEdit.setText(self.temproot_)

    def pushButton_save_click_(self):
        fig1s = self.F1.axes.get_figure()
        print('ipipp')
        fig1s.savefig(self.temproot_ +'\\'+  'lgb.jpg')
        fig2s = self.F2.axes.get_figure()
        fig2s.savefig(self.temproot_ +'\\'+  'xgb.jpg')
        fig3s = self.F3.axes.get_figure()
        fig3s.savefig(self.temproot_  +'\\'+ 'igb_xgb.jpg')
        fig4s = self.F4.axes.get_figure()
        fig4s.savefig(self.temproot_ +'\\'+ '三种结果误差.jpg')

    def xgbplot(self):
        print('yiiyiyiy')
        self.F1.axes.scatter(self.data_xgb.index, self.data_xgb.车主报价预测)
        self.F1.axes.scatter(self.data_xgb.index, self.testdata.loc[self.data_xgb.index]['车主报价（万）'])
        self.F1.axes.set_ylabel('二手车价格')
        # plt.title('折扣率与公里数散点图')
        self.F1.axes.set_title('xgb模型预测结果与真实值差异对比')
        self.F1.axes.tick_params(direction='in')
        self.F1.axes.tick_params(direction='in')

    def lgbtyplot(self):
        print('yiiyiyiy')
        self.F2.axes.scatter(self.data_xgb.index, self.lgb.loc[self.data_xgb.index].车主报价预测)
        self.F2.axes.scatter(self.data_xgb.index, self.testdata.loc[self.data_xgb.index]['车主报价（万）'])
        self.F2.axes.set_ylabel('二手车价格')
        # plt.title('折扣率与公里数散点图')
        self.F2.axes.set_title('lgb模型预测结果与真实值差异对比')
        self.F2.axes.tick_params(direction='in')
        self.F2.axes.tick_params(direction='in')

    def lgbxgb(self):
        self.F3.axes.scatter(self.data_xgb.index, self.lgb_xgb.loc[self.data_xgb.index]['车主报价预测'])
        self.F3.axes.scatter(self.data_xgb.index, self.testdata.loc[self.data_xgb.index]['车主报价（万）'])
        self.F3.axes.set_ylabel('二手车价格')
        # plt.title('折扣率与公里数散点图')
        self.F3.axes.set_title('lgb_xgb融合模型预测结果与真实值差异对比')
        self.F3.axes.tick_params(direction='in')
        self.F3.axes.tick_params(direction='in')

    def piethree(self):
        x1 = self.testdata['车主报价（万）'] - self.xgb.车主报价预测
        x2 = self.testdata['车主报价（万）'] - self.lgb.车主报价预测
        x3 = self.testdata['车主报价（万）'] - self.lgb_xgb.车主报价预测
        print(x1.mean())
        print(x2.mean())
        print(x3.mean())
        labels = [u'xgb模型误差', u'lgb模型误差', u'融合模型误差']
        sizes = [x1.mean(), x2.mean(), x3.mean()]
        self.F4.axes.plot(labels,sizes,'mo:',linewidth=2.5)
        self.F4.axes.set_ylabel('价格误差')
        # plt.title('折扣率与公里数散点图')
        self.F4.axes.set_title('三种模型预测结果与真实值差异对比')
        self.F4.axes.tick_params(direction='in')
        self.F4.axes.tick_params(direction='in')
        self.F1.fig.tight_layout(pad=2.6)


#创建一个matplotlib图形绘制类
class MyFigure(FigureCanvas):
    def __init__(self,width=5, height=4, dpi=100):
        #第一步：创建一个创建Figure
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        #第二步：在父类中激活Figure窗口
        super(MyFigure,self).__init__(self.fig) #此句必不可少，否则不能显示图形
        #第三步：创建一个子图，用于绘制图形用，111表示子图编号，如matlab的subplot(1,1,1)
        self.axes = self.fig.add_subplot(111)



app = QApplication([])
#app = QtWidgets.QApplication(sys.argv)
# MainWindow = QtWidgets.QMainWindow()
SI.loginWin = Main_window()
# print(SI.loginWin.ui.centralwidget)
SI.loginWin.ui.show()
app.exec_()