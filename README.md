# CNN_CIFAR10

## 概述
本Demo是tensorflow利用卷积神经网络实现对CIFAR10进行图像分类，load_data.py用于加载数据，model.py定义了CNN模型，包括 conv1、pool1 && norm1、conv2、pool2 && norm2、full-connect1、full_connect2、softmax。
train.py是用于将模型在CIFAR-10数据集上训练，test.py用于测试模型分类效果。

## 代码说明

1、load_data.py 加载数据集

2、 model.py 模型文件

3、train.py 训练模型

4、test.py 预测模型

5、logs_2目录 模型保存路径

6、data/receive目录 需要预测的图片存放路径

7、cifar_download.py cifar10数据集下载文件

## 运行环境说明

tensorflow 1.X + python3

## 数据集准备

**CIFAR10:** 一共包含 10 个类别的 RGB 彩色图 片：飞机（ a叩lane ）、汽车（ automobile ）、鸟类（ bird ）、猫（ cat ）、鹿（ deer ）、狗（ dog ）、蛙类（ frog ）、马（ horse ）、船（ ship ）和卡车（ truck ）。图片的尺寸为 32×32 ，数据集中一共有 50000 张训练圄片和 10000 张测试图片。 CIFAR-10 含有的是现实世界中真实的物体，不仅噪声很大，而且物体的比例、 特征都不尽相同，这为识别带来很大困难。 直接的线性模型如 Softmax 在 CIFAR-10 上表现得很差。

**离线下载：** 

 ```python
CIFAR_DIR ="D:\\ZJUer\\courses\\datasets\\cifar-10-batches-py\\data_batch_5"#数据集路径
with open(CIFAR_DIR , 'rb') as f:
    data = pickle.load(f, encoding='bytes')
print('----------batch1的基本信息-------------')    
print('data的数据类型：',type(data)) # 输出 <class 'dict'>
print('字典的key名称：',data.keys()) # 输出 dict_keys([b'filenames', b'data', b'labels', b'batch_label'])
print('bdata的数据类型',type(data[b'data'])) # 输出 <class 'numpy.ndarray'>
print('bdata的数据形状',data[b'data'].shape) # 输出 (10000, 3072) 说明有 10000 个样本, 3072个特征
 ```

## 结果展示
### 测试一
![image](https://github.com/CarrollAdmin/CNN_CIFAR10/blob/master/img/predict1.png)
### 测试二
![image](https://github.com/CarrollAdmin/CNN_CIFAR10/blob/master/img/predict2.png)
### 测试三
![image](https://github.com/CarrollAdmin/CNN_CIFAR10/blob/master/img/predict3.png)
