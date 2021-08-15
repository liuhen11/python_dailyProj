# 基于PPYOLO的工业生产流水线金属工件表面缺陷智能检测系统

收集工件表面缺陷的数据集并将其进行VOC数据集格式的划分，并采用PPYOLO算法训练缺陷检测模型，完成检测推理模型的服务器部署工作。

# 一、项目背景

## 1.1  项目背景描述
该项目最初主要来源于实际生产线实习经验，个人专业为机械工程专业，实习时曾参观过工件生产流水线实际运行情况。发现很多时候生产的工件表面会有或多或少不同的缺陷存在，如**裂纹(crazing)、杂质(inclusi-on)、斑块(patches)、点蚀(pitted)、轧入氧化皮(rolled-in scale)与划痕(scratches)** 而这些缺陷又需要工人师傅们在流水线上人工进行挑拣，将其筛选出来。这无疑加大了企业生产的人力成本，也使得生产效率变慢,而采用智能自动检测工件缺陷的方法，可将企业从繁琐重复的劳动中解放出来，提高生产效率。  

目前，表面缺陷检测采用传统的图像处理算法检测具有鲁棒性差，误检率高等缺点，而采用深度学习目标检测的方法可使得检测准确率大大提高，并能同时对缺陷种类进行自动分类。模型的泛化性能和鲁棒性均有显著增强。**飞桨paddlepaadle**框架中，基于yolov3算法改进提出的**ppyolo算法**检测准确率更高，性能更高，故选用该算法模型进行检测系统开发。

## 1.2 项目效果展示
**其工件缺陷检测项目效果如下图所示。**  

![](https://ai-studio-static-online.cdn.bcebos.com/bcbb6739394a4e8386cd8993c6a8ccebcb1312be2ab646a69910a42337b63ff1)



# 二、数据集简介

本次数据集主要用的是开源数据集，来源为东北大学表面缺陷检测论文数据集，数据集地址为：http://faculty.neu.edu.cn/me/songkc/Vision-based_SIS_Steel.html。  

数据集格式为一个zip压缩文件，该数据集已经事先进行了标注，具有对应的标注文件，对数据集处理步骤如下所示：  
## 2.1 数据集解压与VOC数据集格式转换  

```python
# 安装 paddledet 
! pip install paddledet==2.1.0 -i https://mirror.baidu.com/pypi/simple
# 克隆 PaddleDetection 库
! git clone https://gitee.com/paddlepaddle/PaddleDetection.git
# 创建自己的数据文件目录
%cd /home/aistudio/
!pwd
!mv PaddleDetection/ work/
!mkdir work/PaddleDetection/dataset/SteelDEC_VOCData
!unzip -oq /home/aistudio/NEU-DET.zip -d work/PaddleDetection/dataset/SteelDEC_VOCData
  
  
# 初步创建VOC数据格式
!mv work/PaddleDetection/dataset/SteelDEC_VOCData/NEU-DET/Annotations/ work/PaddleDetection/dataset/SteelDEC_VOCData/
!mv work/PaddleDetection/dataset/SteelDEC_VOCData/NEU-DET/JPEGImages/ work/PaddleDetection/dataset/SteelDEC_VOCData/
!rm -r work/PaddleDetection/dataset/SteelDEC_VOCData/NEU-DET
```
```python
# 此时数据集文件目录格式如下所示：
  NEU-DET/
	├── Annotations     # 标注文件
  	├── JPEGImages    # 图像数据

```
```python
#导入paddlex 
!pip install paddlex 
#使用paddlex划分NEU-DET VOC数据集为训练集，验证集，测试集
!paddlex --split_dataset --format VOC --dataset_dir work/PaddleDetection/dataset/SteelDEC_VOCData/ --val_value 0.15  --test_value 0.05
```
```python
此时数据集文件目录格式如下所示：
 NEU-DET/
  ├── Annotations	   # 每张图片相关的标注信息,xml格式
  ├── JPEGImages	   # 包括训练验证测试用到的所有图片  
  ├── train_list.txt	#训练集 
  ├── val_list.txt     # 验证集
  ├── test_list.txt	   # 测试集
  ├── label.txt	      # 标签的类别数
```


## 2.2 数据加载和预处理


```python
# 导入数据处理，模型训练等所需要的库
import paddle
import paddlex as pdx
import numpy as np
import paddle.nn as nn
import paddle.nn.functional as F
import PIL.Image as Image
import cv2 
import os

from random import shuffle
from paddlex.det import transforms as T
from PIL import Image, ImageFilter, ImageEnhance

# 训练数据集
trainDateset = open('work/PaddleDetection/dataset/SteelDEC_VOCData/train_list.txt','r')
trainDateset_jpg = trainDateset.readlines()  

# 验证数据集
evalDateset = open('work/PaddleDetection/dataset/SteelDEC_VOCData/val_list.txt','r')
evalDateset_jpg = evalDateset.readlines()

# 测试数据集
testDateset = open('work/PaddleDetection/dataset/SteelDEC_VOCData/test_list.txt','r')
testDateset_jpg = testDateset.readlines()
print('训练集样本量: {}，验证集样本量: {}，测试集样本量：{}'.format(len(trainDateset_jpg), len(evalDateset_jpg),len(testDateset_jpg)))

# 对数据集进行数据增强等预处理
# 对数据进行增强
def preprocess(dataType="train"):
    if dataType == "train":
        transform = T.Compose([
            #对图像进行mixup操作，模型训练时的数据增强操作，目前仅YOLOv3模型支持该transform
            #T.MixupImage(mixup_epoch=10), 
            #随机扩张图像
            # T.RandomExpand(),
            #以一定的概率对图像进行随机像素内容变换
            # T.RandomDistort(brightness_range=1.2, brightness_prob=0.3), 
            #随机裁剪图像
            T.RandomCrop(), 
            #根据图像的短边调整图像大小
            # T.ResizeByShort(), 
            #调整图像大小,[’NEAREST’, ‘LINEAR’, ‘CUBIC’, ‘AREA’, ‘LANCZOS4’, ‘RANDOM’]
            T.Resize(target_size=608, interp='RANDOM'),  
            #以一定的概率对图像进行随机水平翻转
            T.RandomHorizontalFlip(), 
             #对图像进行标准化
            T.Normalize()                                           
            ])
        return transform
    else:
        transform = T.Compose([
            T.Resize(target_size=608, interp='CUBIC'), 
            T.Normalize()
            ])
        return transform


train_transforms = preprocess(dataType="train")
eval_transforms  = preprocess(dataType="eval")
```

训练集样本量: 1440，验证集样本量: 270，测试集样本量：90


## 2.3 数据集查看


```python
print('图片：')
print(type(trainDateset_jpg[0]))
print(trainDateset_jpg[0])
label = open('work/PaddleDetection/dataset/SteelDEC_VOCData/labels.txt','r')
label_ = label.readlines()
print('标签类型：')
print(label_)

#可视化展示
import matplotlib.pyplot as plt
plt.figure()
im = plt.imread('work/PaddleDetection/dataset/SteelDEC_VOCData/JPEGImages/crazing_51.jpg')
plt.imshow(im)
plt.show()

```
图片：  

<class 'str'>
JPEGImages/inclusion_81.jpg Annotations/inclusion_81.xml

标签类型：  

['crazing\n', 'inclusion\n', 'patches\n', 'pitted_surface\n', 'rolled-in_scale\n', 'scratches\n']  

![](https://ai-studio-static-online.cdn.bcebos.com/2d3b98b7e9b04aa88076767a86d7bc0563f11bb704d74a5ab8a1b54abf04e180)  
## 2.4 定义所使用的训练集和验证集
```python
%cd /home/aistudio/work/PaddleDetection

# 定义训练和验证所用的数据集
# API地址：https://paddlex.readthedocs.io/zh_CN/develop/data/format/detection.html?highlight=paddlex.det
train_dataset = pdx.datasets.VOCDetection(
    data_dir='./dataset/SteelDEC_VOCData',
    file_list='./dataset/SteelDEC_VOCData/train_list.txt',
    label_list='./dataset/SteelDEC_VOCData/labels.txt',
    transforms=train_transforms,
    shuffle=True)
eval_dataset = pdx.datasets.VOCDetection(
    data_dir='./dataset/SteelDEC_VOCData',
    file_list='./dataset/SteelDEC_VOCData/val_list.txt',
    label_list='./dataset/SteelDEC_VOCData/labels.txt',
    transforms=eval_transforms)
```


# 三、模型选择和开发

本次算法模型选择基于**PPYOLO**进行训练部署开发。**PPYOLO**是基于yolo v3算法模型进行改进，加入了好多模型优化的技巧形成的，并且对于小目标甚至实际部署的效果要好于yolo v3以及yolo v4等。个人有幸使用过P-PYOLO模型训练自己的数据集，最大的感受就是接口易用，训练简单，效果且很好，如下图所示为PPYOLO算法与其他目标检测算法的论文结果实验对比：  
![](https://ai-studio-static-online.cdn.bcebos.com/c74744fc33474c8d9f3c1f4c9540ac8c25dac22cc3164f2590f93b69065b48e1)  
可以看出，**PPYOLO**在FPS更高的情况下，仍能保持较好的map精度值，这无疑依赖于论文研究作者对yolo v3良好的backbone以及其余tricks的使用。**PPYOLO**论文地址为：https://arxiv.org/abs/2007.12099。下面对**PP-YOLO**算法模型作理论的解释与项目的开发运用。


## 3.1 模型网络结构

![](https://ai-studio-static-online.cdn.bcebos.com/4f9d9a1f9e87455aa09ef398fc3b8520c6c04407537f4e8bbdfabe2ce40d8d38)

如上图所示，PPYOLO算法模型采用的Resnet作为网络的backbone提取图像特征，同时为了避免精度的下降，采用了DCN的可变形卷积结构，这样使得backbone提取的图像精度较好，训练得到的精度较高。而NECK层使用FPN特征金字塔结构。

**模型网络结构搭建**
模型网络结构搭建细节请具体参考**Paddlepaddle Detecion**相关套件，本次主要是基于PPYOLO模型进行应用开发。

## 3.2 模型改进技术点

**1.** PPYOLO论文中将BAtchsize从64提升到196，这虽然对于机器的算力硬件配置要求更高，但在算力充足的情况下，采用较大且合理的Batchsize可以提高训练的稳定性，加速模型的收敛，得到较好的结果。  
  
**2.** 使用了滑动平均窗口（EMA）的方法更新模型训练中的参数，移动平均可使模型训练的参数更新更加稳定,这样会使模型较容易且收敛结果更加稳定可靠，而不是一种看似收敛但其实还未收敛的状态，至于具体为什么该方法可以使参数更新更加稳定以及拟合效果，请读者参考原论文以及相关论文的相关部分，发散自己的思维进行合理思考解释。  

**3.** 在Head层中，PPYOLO将特征图上随意抛弃的几块连续区域（Drop Block）进行置零操作。  

**4.** IOU损失的改进，在L1损失相同的情况下，不同的交并比得到的真实框与预测框的交集面积也不同。L1损失的对四个坐标点独立的计算损失，没有考虑到他们作为一个锚框的整体。其次L1损失并不具备尺度不变性。因此统一评价指标和损失函数对于训练定位框回归有较好的作用，PPYOLO增加一个分支计算IoU损失.细节如下图所示：（参考图链接：https://blog.csdn.net/forewill/article/details/107768766）  
![](https://ai-studio-static-online.cdn.bcebos.com/bd62a163dca141559ee7c304115b9c6ca287bf66b1274a069e2a989063b6348e)  
Iou损失的公式为IoU loss = -ln(Intersection/Union)，即负的两框相交的面积除以相并的面积取对数。  

**5.** yolo v3的一个缺陷就是，当所预测的box的坐标点落在单元格边框上，则需要将Px，Py预测的足够大才能拟合目标点，不平滑的Px，Py不利于网络的训练及测试，需要有个缩放因子将其扩大，缓解这种情况。因此PPYOLO提出了如下公式来克服边界敏感情况：  
![](https://ai-studio-static-online.cdn.bcebos.com/6844d873c4e6419faff43d9ef824b641d7d93ab734ea42b3819b8fcc156ab851)  

**6.** PPYOLO中预测后处理时将非极大值抑制替换为Matrix NMS。Matrix NMS将mask IoU并行化是最主要的一个提升作用和加速作用。  

**7.** SPP层增加可以扩大图像提取特征的感受野，PPYOLO引入{1, 5, 9, 13}这几种大小的最大池化减少了部分了计算量。   

**8.** 使用更好的预训练模型可以让模型的效果更棒，PPYOLO使用了蒸馏的ResNet50-vd作为与训练模型。

```python
# 使用paddlex模块进行PPYOLO模型定义
model = pdx.det.PPYOLO(num_classes=num_classes, )

```


## 3.3 模型训练


```python
# 完整训练代码
# ...
# ...
import matplotlib
matplotlib.use('Agg') 
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # 配置显卡
%matplotlib inline
import warnings
warnings.filterwarnings("ignore")

#num_classes有些模型需要加1 比如faster_rcnn
num_classes = len(train_dataset.labels)

# 定义PPYOLO模型
model = pdx.det.PPYOLO(num_classes=num_classes, )

# 开始进行训练
model.train(
    num_epochs=140,                     # 设置训练轮数
    train_dataset=train_dataset,            # 设置训练数据集
    train_batch_size=16,                  # 设置Bs
    eval_dataset=eval_dataset,             # 设置评估验证数据集
    learning_rate=3e-5,                  # 设置学习率
    warmup_steps=90,                   
    warmup_start_lr=0.0,
    # 定义保村间隔轮数，即每7轮保存一次训练的模型结果
    save_interval_epochs=7,               
    lr_decay_epochs=[42, 70],             # 定义学习率衰减轮数范围
    save_dir='output/PPYOLO',             # 定义模型保存输出文件目录
    # 定义预训练模型参数文件
    pretrain_weights='work/PaddleDetection/output/PPYOLO/epoch_28/model.pdparams',      
    use_vdl=True)                     # 定义使用Visual DL可视化工具
```
    2021-08-01 16:33:40 [INFO]	[TRAIN] Epoch=1/140, Step=2/90, loss=13136.869141, lr=0.0, time_each_step=5.63s, eta=20:13:3
    2021-08-01 16:33:42 [INFO]	[TRAIN] Epoch=1/140, Step=4/90, loss=10573.105469, lr=1e-06, time_each_step=3.3s, eta=11:51:10
    2021-08-01 16:33:43 [INFO]	[TRAIN] Epoch=1/140, Step=6/90, loss=4083.589844, lr=2e-06, time_each_step=2.44s, eta=8:46:28
    2021-08-01 16:33:45 [INFO]	[TRAIN] Epoch=1/140, Step=8/90, loss=3452.110596, lr=2e-06, time_each_step=2.03s, eta=7:17:55
    2021-08-01 16:33:47 [INFO]	[TRAIN] Epoch=1/140, Step=10/90, loss=2868.198242, lr=3e-06, time_each_step=1.79s, eta=6:24:43
    2021-08-01 16:33:48 [INFO]	[TRAIN] Epoch=1/140, Step=12/90, loss=1248.065186, lr=4e-06, time_each_step=1.63s, eta=5:51:7
    2021-08-01 16:33:50 [INFO]	[TRAIN] Epoch=1/140, Step=14/90, loss=311.386871, lr=4e-06, time_each_step=1.52s, eta=5:27:1
    2021-08-01 16:33:52 [INFO]	[TRAIN] Epoch=1/140, Step=16/90, loss=338.506287, lr=5e-06, time_each_step=1.47s, eta=5:17:3
    2021-08-01 16:33:54 [INFO]	[TRAIN] Epoch=1/140, Step=18/90, loss=125.094307, lr=6e-06, time_each_step=1.4s, eta=5:2:13
    2021-08-01 16:33:56 [INFO]	[TRAIN] Epoch=1/140, Step=20/90, loss=69.620895, lr=6e-06, time_each_step=1.36s, eta=4:53:27
    2021-08-01 16:33:58 [INFO]	[TRAIN] Epoch=1/140, Step=22/90, loss=86.43985, lr=7e-06, time_each_step=0.92s, eta=3:18:9
    2021-08-01 16:34:00 [INFO]	[TRAIN] Epoch=1/140, Step=24/90, loss=58.780914, lr=8e-06, time_each_step=0.89s, eta=3:11:10
    2021-08-01 16:34:01 [INFO]	[TRAIN] Epoch=1/140, Step=26/90, loss=34.087837, lr=8e-06, time_each_step=0.88s, eta=3:9:20
    2021-08-01 16:34:03 [INFO]	[TRAIN] Epoch=1/140, Step=28/90, loss=45.55809, lr=9e-06, time_each_step=0.89s, eta=3:11:31
    2021-08-01 16:34:04 [INFO]	[TRAIN] Epoch=1/140, Step=30/90, loss=49.472973, lr=1e-05, time_each_step=0.89s, eta=3:10:42
    2021-08-01 16:34:06 [INFO]	[TRAIN] Epoch=1/140, Step=32/90, loss=40.846176, lr=1e-05, time_each_step=0.88s, eta=3:10:21
    2021-08-01 16:34:08 [INFO]	[TRAIN] Epoch=1/140, Step=34/90, loss=42.284126, lr=1.1e-05, time_each_step=0.89s, eta=3:10:22
    2021-08-01 16:34:10 [INFO]	[TRAIN] Epoch=1/140, Step=36/90, loss=52.995316, lr=1.2e-05, time_each_step=0.88s, eta=3:9:14
    
    



## 3.4 模型预测

### 3.4.1 批量预测

使用model.predict接口来完成对大量数据集的批量预测。


```python
# 使用paddlex加载训练过程中保存的最好的训练模型
model = pdx.load_model('output/PPYOLO/best_model')

# 定义测试集数据文件路径
image_dir_file = './dataset/SteelDEC_VOCData/test_list.txt'
import pandas as pd
# 读取测试集txt文件，获取测试集每张图片的文件路径
txt_png = pd.read_csv(image_dir_file,header=None)
txt_jpgs = txt_png.iloc[:,0].str.split(' ',expand=True)[0].tolist()
txt_jpgs
# images = os.listdir(image_dir)

# 遍历每张测试集图片，进行预测
for img in txt_jpgs:
    image_name = './dataset/SteelDEC_VOCData/' + img
    result = model.predict(image_name)             # 使用predict接口进行预测
    pdx.det.visualize(image_name, result, threshold=0.2, save_dir='./output/PPYOLO/img_predict')
    # 设定阈值，将预测的图片保存到img_predict目录下
```

    Predict begin...
    2021-08-01 11:47:42 [INFO]	Model[PPYOLO] loaded.
    2021-08-01 11:47:42 [INFO]	The visualized result is saved as ./output/PPYOLO/img_predict/visualize_crazing_256.jpg
    2021-08-01 11:47:42 [INFO]	The visualized result is saved as ./output/PPYOLO/img_predict/visualize_scratches_66.jpg
    2021-08-01 11:47:42 [INFO]	The visualized result is saved as ./output/PPYOLO/img_predict/visualize_pitted_surface_232.jpg
    2021-08-01 11:47:42 [INFO]	The visualized result is saved as ./output/PPYOLO/img_predict/visualize_crazing_16.jpg
    2021-08-01 11:47:42 [INFO]	The visualized result is saved as ./output/PPYOLO/img_predict/visualize_pitted_surface_206.jpg
    2021-08-01 11:47:43 [INFO]	The visualized result is saved as ./output/PPYOLO/img_predict/visualize_scratches_152.jpg
    2021-08-01 11:47:43 [INFO]	The visualized result is saved as ./output/PPYOLO/img_predict/visualize_crazing_240.jpg
    2021-08-01 11:47:43 [INFO]	The visualized result is saved as ./output/PPYOLO/img_predict/visualize_scratches_245.jpg
    2021-08-01 11:47:43 [INFO]	The visualized result is saved as ./output/PPYOLO/img_predict/visualize_scratches_49.jpg
    2021-08-01 11:47:43 [INFO]	The visualized result is saved as ./output/PPYOLO/img_predict/visualize_inclusion_90.jpg
    2021-08-01 11:47:43 [INFO]	The visualized result is saved as ./output/PPYOLO/img_predict/visualize_rolled-in_scale_242.jpg
    2021-08-01 11:47:43 [INFO]	The visualized result is saved as ./output/PPYOLO/img_predict/visualize_pitted_surface_278.jpg
    2021-08-01 11:47:43 [INFO]	The visualized result is saved as ./output/PPYOLO/img_predict/visualize_crazing_4.jpg
    2021-08-01 11:47:43 [INFO]	The visualized result is saved as ./output/PPYOLO/img_predict/visualize_scratches_183.jpg
    2021-08-01 11:47:43 [INFO]	The visualized result is saved as ./output/PPYOLO/img_predict/visualize_rolled-in_scale_173.jpg
    2021-08-01 11:47:43 [INFO]	The visualized result is saved as ./output/PPYOLO/img_predict/visualize_inclusion_64.jpg
    2021-08-01 11:47:43 [INFO]	The visualized result is saved as ./output/PPYOLO/img_predict/visualize_crazing_85.jpg
    2021-08-01 11:47:43 [INFO]	The visualized result is saved as ./output/PPYOLO/img_predict/visualize_patches_246.jpg
    2021-08-01 11:47:43 [INFO]	The visualized result is saved as ./output/PPYOLO/img_predict/visualize_scratches_141.jpg
    2021-08-01 11:47:43 [INFO]	The visualized result is saved as ./output/PPYOLO/img_predict/visualize_pitted_surface_164.jpg
    2021-08-01 11:47:44 [INFO]	The visualized result is saved as ./output/PPYOLO/img_predict/visualize_inclusion_127.jpg
    2021-08-01 11:47:44 [INFO]	The visualized result is saved as ./output/PPYOLO/img_predict/visualize_pitted_surface_239.jpg
    ......


### 3.4.2 单张图片预测可视化结果

对单张图片进行预测结果可视化。


```python
#展示模型推理结果
import PIL.Image as Image
import matplotlib.pyplot as plt
%cd /home/aistudio/work/PaddleDetection
path = "./dataset/SteelDEC_VOCData/JPEGImages/inclusion_268.jpg"
img = Image.open(path)
plt.imshow(img)          #根据数组绘制图像
plt.show()               #显示图像

path = 'output/PPYOLO/img_predict/visualize_inclusion_268.jpg'
img = Image.open(path)
plt.imshow(img)          #根据数组绘制图像
plt.show()               #显示图像
```
![](https://ai-studio-static-online.cdn.bcebos.com/575eac2097c04b889ce166c45669e18c20ff2aa2e3e641e6a0786b13f44e767b)

# 四、使用paddlehub模型进行服务器端部署  
## 4.1 模型部署

使用paddlex将训练模型进行导出，进行量化解析为推理模型。
```python
!paddlex --export_inference --model_dir=work/PaddleDetection/output/PPYOLO/best_model/ --save_dir=./inference_model
```
导出后的推理模型文件夹为inference_model，一共有三个文件，__params__，__model__，model.yml

基于上述安装好的paddlex模块，使用paddlehub将上述inference_model转换为hub模型。
```python
!hub convert --model_dir inference_model \
              --module_name Default_Steel \
              --module_version 1.0.0 \
              --output_dir output

# 转化好的模型将保存在output目录中，并成为一个Default_Steel.tar.gz的压缩包文件
# 使用hub将该模型安装好
!hub install output/Default_Steel.tar.gz
```
Decompress output/Default_Steel.tar.gz
[##################################################] 100.00%  
[2021-08-08 20:50:25,564] [    INFO] - Successfully uninstalled Default_Steel  
[2021-08-08 20:50:25,914] [    INFO] - Successfully installed Default_Steel-1.0.0 

使用如下命令行进行模型部署
$ hub serving start --modules/-m [Module1==Version1, Module2==Version2, ...] \
                    --port/-p XXXX
                    --config/-c XXXX
                    
| 参数 |	用途 |
| ----------| ----------|
| --modules/-m |	PaddleHub Serving预安装模型，以多个Module==Version键值对的形式列出。当不指定Version时，默认选择最新版本 |
| --port/-p |	服务端口，默认为8866 |
| --config/-c |	使用配置文件配置模型 |

因此，我们仅需要一行代码即可完成模型的部署，如下（**注：AIStudio上要再终端运行**）：

	$ hub serving start -m Default_Steel
        
![](https://ai-studio-static-online.cdn.bcebos.com/f867361a407043ed9234b08e3c54052f2f0e697bbf1e4b7ebbf496c81dc201e7)



等待模型加载后，此预训练模型就已经部署在机器上了。  
在第二步模型安装的同时，会生成一个客户端请求示例，存放在模型安装目录，默认为${HUB_HOME}/.paddlehub/modules，对于此例，我们可以在~/.paddlehub/modules/Default_Steel找到此客户端示例serving_client_demo.py，代码如下

这里的路径为：/home/aistudio/.paddlehub/modules/Default_Steel/serving_client_demo.py （将图片路径进行了修改）,具体修改方式如下：  
**1.** 在AIStudio终端，将路径进行切换，使用vim serving_client_demo.py进入文本编辑器模式，根据需求改变模型图片路径，便可进行部署模型测试。如下图所示：  
![](https://ai-studio-static-online.cdn.bcebos.com/02be3674700c4022801b6f0e601ddd338d9e5452923048af95883285287a9470)
**2.** 将部署在Aistudio终端服务器的模型进行部署测试，具体命令如下所示：  
```python
!python /home/aistudio/.paddlehub/modules/Default_Steel/serving_client_demo.py
```  
其返回响应结果如下所示：  
/home/aistudio/.paddlehub/modules/Default_Steel/serving_client_demo.py:10: DeprecationWarning: tostring() is deprecated. Use tobytes() instead.
  return base64.b64encode(data.tostring()).decode('utf8')  

[[{'bbox': [134.1076202392578, 0.0, 34.68701171875, 200.0], 'category': 'scratches', 'category_id': 5, 'score': 0.5157753229141235}, {'bbox': [128.9130859375, 0.0, 31.07855224609375, 200.0], 'category': 		'scratches', 'category_id': 5, 'score': 0.13788828253746033}, {'bbox': [128.05682373046875, 0.0, 33.906341552734375, 180.96328735351562], 'category': 'scratches', 'category_id': 5, 'score': 					0.12424053251743317}, {'bbox': [129.0009307861328, 24.918968200683594, 30.1689453125, 175.0810317993164], 'category': 'scratches', 'category_id': 5, 'score': 0.08536934852600098}, {'bbox': 					[126.94180297851562, 3.0034637451171875, 33.327423095703125, 196.9965362548828], 'category': 'scratches', 'category_id': 5, 'score': 0.08183740824460983}, {'bbox': [125.47954559326172, 						6.057464599609375, 39.10143280029297, 147.8684539794922], 'category': 'scratches', 'category_id': 5, 'score': 0.07568959891796112}, {'bbox': [144.8748016357422, 0.0, 29.405029296875, 200.0], 'category': 		  'scratches', 'category_id': 5, 'score': 0.06888245046138763}, {'bbox': [146.51556396484375, 9.404808044433594, 25.298919677734375, 190.5951919555664], 'category': 'scratches', 'category_id': 5, 				'score': 0.06098642200231552}, {'bbox': [134.0897674560547, 5.3186187744140625, 35.152587890625, 149.18048095703125], 'category': 'scratches', 'category_id': 5, 'score': 0.058005478233098984}, {'bbox': 		  [116.36105346679688, 9.472007751464844, 38.328155517578125, 179.35746002197266], 'category': 'scratches', .......

# 五、效果展示

## 5.1 项目运行方法。  
**1.** 首先安装好模型训练以及数据处理等常用的库。  
**2.** 将上述项目fork为自己的项目，然后执行上述代码即可。  
**3.** 将项目中用到的数据集，替换成自己的数据集，并修改相应的数据路径以及训练模型路径，自定义就行，然后接着就愉快的炼丹就行。  
## 5.2 项目运行效果  
![](https://ai-studio-static-online.cdn.bcebos.com/6bb26a60119047b88e2deb9cb5a43e4231984c0e1b284b16b11651a279637583)


项目目前完成了模型的训练，调参，模型的推理导出，map也达到了45.3左右，精度尚可，并且完成了推理模型的服务器端部署和测试，具体效果展示如上述第四节图文所示。

# 六、总结与升华
## 6.1 总结
**1.** 此次模型训练部署整体来说过程较为顺利，但在做项目的过程中，或多或少也遇到了不少问题，采用不同方法进行了解决。  
**2.** 首先对于某些接口的不熟悉，导致程序代码运行总是报错，通过查阅paddle paddle detection以及paddlex相关的文档进行解决。  
**3.** 此外就是使用paddlehub进行模型部署时遇到了某些坑，通过助教的答疑解惑，自己进行多次尝试最终得到解决。  
## 6.2 升华  
**1.** 此次项目由于时间仓促，且条件有限，数据集采用的是开源的数据集，实际运用肯定够呛，本次主要重在目标检测模型的demo工程，可采用该方法在更好的数据集上进行移植。  
**2.** 由于数据集缺乏，模型的鲁棒性和泛化性必然很差，未来可到实际生产工业流水线进行数据的采集（如高速，低速，强光，弱光等不同环境下），作进一步的模型训练部署开发，使模型适应实际生产场景，提高模型的鲁棒性。  
**3.** 可考虑进行模型的拓展，例如多模型融合以提升模型的精度和泛化等。  
**4.** 可在实际生产线采用边缘计算设备进行部署拓展等。

**通过本次项目学到了CV目标检测领域模型的训练，预测，推理，部署，数据的预处理增强等，完成了全流程的demo操作，有好的地方，也有不足之处。总之就是，革命尚未成功，同志仍需努力。**

# 个人简介

> 马上毕业的小硕一枚，百度飞桨小粉丝兼非科班CV爱好者以及小白一枚

> 感兴趣的方向为：CV领域

> Aistudio链接为：(https://aistudio.baidu.com/aistudio/personalcenter/thirdview/354314)

> 也欢迎大家fork、评论交流，互关。
