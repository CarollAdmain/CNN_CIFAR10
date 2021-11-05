# -*- coding: utf-8 -*-
import os
import shutil
import tensorflow as tf
import numpy as np
import time
#import load_data
#import model
from load_data import *
from model import *
import matplotlib.pyplot as plt
import sys
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from matplotlib import rcParams
rcParams['font.family'] = 'simhei'
import matplotlib
matplotlib.rcParams['axes.unicode_minus']=False
#global_iii=0


# 训练模型
def training():
    N_CLASSES = 3
    IMG_SIZE = 208
    BATCH_SIZE = 64
    CAPACITY = 2000
    MAX_STEP = 500
    LEARNING_RATE = 1e-4

    # 测试图片读取
    image_dir = 'D:\\ZJUer\\courses\\cifar10_data\\raw'
    logs_dir = 'logs_2'     # 检查点保存路径A

    sess = tf.Session()

    loss_vec = []
    acc_vec = []
    
    train_list = get_all_files(image_dir, True)
    image_train_batch, label_train_batch = get_batch(train_list, IMG_SIZE, BATCH_SIZE, CAPACITY, True)
    train_logits = inference(image_train_batch, N_CLASSES)
    train_loss = losses(train_logits, label_train_batch)
    train_acc = evaluation(train_logits, label_train_batch)

    train_op = tf.train.AdamOptimizer(LEARNING_RATE).minimize(train_loss)

    var_list = tf.trainable_variables()
    paras_count = tf.reduce_sum([tf.reduce_prod(v.shape) for v in var_list])
    print('参数数目:%d' % sess.run(paras_count), end='\n\n')

    saver = tf.train.Saver()

    sess.run(tf.global_variables_initializer())

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    s_t = time.time()
    try:
        for step in range(MAX_STEP):
            if coord.should_stop():
                break

            _, loss, acc = sess.run([train_op, train_loss, train_acc])
            loss_vec.append(loss)
            acc_vec.append(acc)
            
            if step % 10 == 0:  # 实时记录训练过程并显示
                runtime = time.time() - s_t
                print('Step: %6d, loss: %.8f, accuracy: %.2f%%, time:%.2fs, time left: %.2fhours'
                      % (step, loss, acc * 100, runtime, (MAX_STEP - step) * runtime / 360000))
                s_t = time.time()

            if step % 100 == 0 or step == MAX_STEP - 1:  
                checkpoint_path = os.path.join(logs_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)

    except tf.errors.OutOfRangeError:
        print('Done.')
    finally:
        coord.request_stop()
    
    plt.plot(loss_vec, 'b')
    plt.title("loss")
    plt.xlabel("迭代次数")
    plt.ylabel("loss")
    plt.show()
    
    plt.plot(acc_vec, 'b--', label="train")
    plt.title("准确率曲线")
    plt.xlabel("迭代次数")
    plt.ylabel("准确率")
    plt.legend()
    plt.show()
    
    coord.join(threads=threads)
    sess.close()
    

#%%

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
        




if __name__ == '__main__':
    training()

