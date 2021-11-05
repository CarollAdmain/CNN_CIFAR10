# -*- coding: utf-8 -*-
"""
Created on Fri Oct  8 10:51:34 2021

@author: 57374
"""

import pickle
import matplotlib.pyplot as plt

CIFAR_DIR ="D:\\ZJUer\\courses\\datasets\\cifar-10-batches-py\\data_batch_5"#数据集路径
with open(CIFAR_DIR , 'rb') as f:
    data = pickle.load(f, encoding='bytes')

print('----------batch1的基本信息-------------')    
print('data的数据类型：',type(data)) # 输出 <class 'dict'>
print('字典的key名称：',data.keys()) # 输出 dict_keys([b'filenames', b'data', b'labels', b'batch_label'])
print('bdata的数据类型',type(data[b'data'])) # 输出 <class 'numpy.ndarray'>
print('bdata的数据形状',data[b'data'].shape) # 输出 (10000, 3072) 说明有 10000 个样本, 3072个特征

label0=4100
label1=4100
label2=4100
label3=4100
label4=4100
label5=4100
label6=4100
label7=4100
label8=4100
label9=4100

for index in range(0,9999):
#index=4#打印第几张图片
    print('-----------第%d张图片信息----------'%index)
    print('filenames:',data[b'filenames'][index])
    print('labels:',data[b'labels'][index])
    #print('batch_label:',data[b'batch_label'][index])
    image_arr = data[b'data'][index] # 拿出 第 index 个样本
    image_arr = image_arr.reshape((3, 32, 32)) # 将一维向量改变形状得到这样一个元组:(高,宽,通道数)
    image_arr = image_arr.transpose((1, 2, 0)) 
    plt.imshow(image_arr) # 输出图片
    label = data[b'labels'][index]
    if label == 0:
        label0 = label0 + 1
        plt.savefig("D:\\ZJUer\\courses\\cifar10_data\\raw10\\0_%d.jpg"%label0)#保存图片
        plt.show()
    elif label == 1:
        label1 = label1 + 1
        plt.savefig("D:\\ZJUer\\courses\\cifar10_data\\raw10\\1_%d.jpg"%label1)#保存图片
        plt.show()
    elif label == 2:
        label2 = label2 + 1
        plt.savefig("D:\\ZJUer\\courses\\cifar10_data\\raw10\\2_%d.jpg"%label2)#保存图片
        plt.show()
    elif label == 3:
        label3 = label3 + 1
        plt.savefig("D:\\ZJUer\\courses\\cifar10_data\\raw10\\3_%d.jpg"%label3)#保存图片
        plt.show()
    elif label == 4:
        label4 = label4 + 1
        plt.savefig("D:\\ZJUer\\courses\\cifar10_data\\raw10\\4_%d.jpg"%label4)#保存图片
        plt.show()
    elif label == 5:
        label5 = label5 + 1
        plt.savefig("D:\\ZJUer\\courses\\cifar10_data\\raw10\\5_%d.jpg"%label5)#保存图片
        plt.show()
    elif label == 6:
        label6 = label6 + 1
        plt.savefig("D:\\ZJUer\\courses\\cifar10_data\\raw10\\6_%d.jpg"%label6)#保存图片
        plt.show()
    elif label == 7:
        label7 = label7 + 1
        plt.savefig("D:\\ZJUer\\courses\\cifar10_data\\raw10\\7_%d.jpg"%label7)#保存图片
        plt.show()
    elif label == 8:
        label8 = label8 + 1
        plt.savefig("D:\\ZJUer\\courses\\cifar10_data\\raw10\\8_%d.jpg"%label8)#保存图片
        plt.show()
    elif label == 9:
        label9 = label9 + 1
        plt.savefig("D:\\ZJUer\\courses\\cifar10_data\\raw10\\9_%d.jpg"%label9)#保存图片
        plt.show()
    
         
    