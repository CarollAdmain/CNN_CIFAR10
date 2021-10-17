# -*- coding: utf-8 -*-
import os
import shutil
import tensorflow as tf
import numpy as np
import time
import pyttsx3
import threading

import socket
import sys
import struct

from load_data import *
from model import *
import matplotlib.pyplot as plt
import sys
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import chardet
import codecs
lock=threading.Lock()
start=time.time()

class MyDirEventHandler(FileSystemEventHandler):
    global global_iii
    def on_moved(self, event):
        print(event)
        eval()               
    def on_created(self, event):
        print(event)
    def on_deleted(self, event):
        print(event)        
    def on_modified(self, event):
        print("modified:", event)
        eval()


        
# 测试检查点
def eval():
        
        tf.reset_default_graph()
        N_CLASSES = 3
        IMG_SIZE = 208
        BATCH_SIZE = 1
        CAPACITY = 200
        MAX_STEP = 1
     
        
        test_dir = 'D:\\ZJUer\\courses\\CNN_Model\\data\\receive'
        logs_dir = 'logs_2'     # 检查点目录
        path=test_dir
        sess = tf.Session()
    
    
        
        i=1
    #对目录下的文件进行遍历
        for file in os.listdir(path):
            if os.path.isfile(os.path.join(path,file))==True:
    #设置新文件名
                new_name=file.replace(file,"0_%d.jpg"%i)
    #重命名
            os.rename(os.path.join(path,file),os.path.join(path,new_name))
            i+=1
    #结束
            
        train_list = get_all_files(test_dir, is_random=True)
        image_train_batch, label_train_batch = get_batch(train_list,IMG_SIZE, BATCH_SIZE, CAPACITY, True)
        train_logits = inference(image_train_batch, N_CLASSES)
        train_logits = tf.nn.softmax(train_logits)  # 用softmax转化为百分比数值
    
        # 载入检查点
        saver = tf.train.Saver()
        print('\n载入检查点...')
        ckpt = tf.train.get_checkpoint_state(logs_dir)
        if ckpt and ckpt.model_checkpoint_path:
            global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
            saver.restore(sess, ckpt.model_checkpoint_path)
            print('载入成功，global_step = %s\n' % global_step)
        else:
            print('没有找到检查点')
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    
        try:
            for step in range(MAX_STEP):
                if coord.should_stop():
                    break
    
                image, prediction = sess.run([image_train_batch, train_logits])
                max_index = np.argmax(prediction)
 
    
                if max_index == 0:

                    answer="airplane"
                    print(answer)
                    plt.imshow(image[0])
                    plt.show()

                    
                elif max_index == 1:

                    answer="automobile"
                    print(answer)
    

                    plt.imshow(image[0])
                    plt.show()
                    
                elif max_index == 2:

                    answer="bird"
                    print(answer)

                    plt.imshow(image[0])
                    plt.show()
    
        except tf.errors.OutOfRangeError:
            print('Done.')
        finally:
            coord.request_stop()
    
        coord.join(threads=threads)
        #删除文件
        filelist=[]                      #选取删除文件夹的路径,最终结果删除img文件夹
        filelist=os.listdir(test_dir)                #列出该目录下的所有文件名
        for f in filelist:
            filepath = os.path.join( test_dir, f )   #将文件名映射成绝对路劲
            if os.path.isfile(filepath):            #判断该文件是否为文件或者文件夹
                os.remove(filepath)                 #若为文件，则直接删除
                print(str(filepath)+" removed!")
            elif os.path.isdir(filepath):
                shutil.rmtree(filepath,True)        #若为文件夹，则删除该文件夹及文件夹内所有文件
                print("dir "+str(filepath)+" removed!")
        tf.reset_default_graph()
            
        sess.close()



if __name__ == '__main__':  
    
    test_dir = 'D:\\guoshushibie\\data\\receive'
    logs_dir = 'logs_2'     # 检查点目录
    path=test_dir
    eval()

    
