# CNN_CIFAR10

## 概述
本Demo是基于tensorflow利用卷积神经网络实现对CIFAR10的图像分类，load_data.py用于加载数据，model.py定义了CNN模型，包括 conv1、pool1 && norm1、conv2、pool2 && norm2、full-connect1、full_connect2、softmax。train.py是用于将模型在CIFAR-10数据集上训练，test.py用于测试模型分类效果。

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

**CIFAR10:** 一共包含 10 个类别的 RGB 彩色图 片，数据集中一共有 50000 张训练圄片和 10000 张测试图片。 CIFAR-10 含有的是现实世界中真实的物体，不仅噪声很大，而且物体的比例、 特征都不尽相同，这为识别带来很大困难。 直接的线性模型如 Softmax 在 CIFAR-10 上表现得很差。

**离线下载：** 

设置cifar10离线下载路劲，运行cifar_download.py,会以X_Y.png格式保存到预设的路径,X为图片类别，Y为图片编号
```python
python cifar_download.py
 ```
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
 
## 模型构建

model.py定义了CNN模型，包括 conv1、pool1 && norm1、conv2、pool2 && norm2、full-connect1、full_connect2、softmax。

提供接口interface(image, n_classes),支持多分类图片训练与预测。

```python
 def inference(images, n_classes)：
 ```
 
## 模型训练

trainy.py定义模型训练过程，load_data.py加载数据集中的图片，并且加载到的图片是乱序的。迭代过程中通过保存每一次迭代的临时loss和准确率，每100次迭代和最后一次迭代时，保存模型。模型训练完毕绘制loss曲线和准确率曲线，对模型进行评估。

```python
python train.py
 ```
 
**运行过程展示：** 

 
 
 **效果评估：**
 
 
 
 ## 图片分类
 
 test.py利用模型训练过程中得到模型，对CIFAR10测试集中图片进行分类。使用者将需要预测分类的图片存放在data/receive目录下，运行test.py,即可得出结果：当前需要预测分类的图片，以及它的分类。
 
 ```python
 1、将需预测分类的图片存放在对应目录
 2、python test.py
 ```

## 结果展示
### 测试一
![image](https://github.com/CarrollAdmin/CNN_CIFAR10/blob/master/img/predict1.png)
### 测试二
![image](https://github.com/CarrollAdmin/CNN_CIFAR10/blob/master/img/predict2.png)
### 测试三
![image](https://github.com/CarrollAdmin/CNN_CIFAR10/blob/master/img/predict3.png)
