# -*- coding: utf-8 -*-
"""
Created on Sun Mar 31 19:26:18 2019

@author: GEORGE
"""

import tensorflow as tf
from PIL import Image
from code import cuts
import shutil
import os
image_size = 28 
results = []
def run (url):
    row = 1
    col = 4
    if row > 0 and col > 0:
        cuts.splitimage(url, row, col)
    else:
        print('无效的行列切割参数！')
    files1 = os.listdir("./cutImage")
    sess=tf.Session()
    saver = tf.train.import_meta_graph('./cnn_model/cnn_model.ckpt.meta')
    saver.restore(sess, "./cnn_model/cnn_model.ckpt")
    for i in files1:
        img = Image.open("./cutImage/"+i)
        s = img.resize((28,28))
        s = tf.reshape(s, [1, image_size, image_size, 1])
        s = tf.cast(s, tf.float32) * (1. / 255.) - 0.5
        #先加载图和参数变量
        graph = tf.get_default_graph()
        w1 = graph.get_tensor_by_name("images:0")
        w2 = graph.get_tensor_by_name("Fc2/softmax/my_prediction:0")
        w3 = graph.get_tensor_by_name("Fc1/my_keep_prob:0")
        s = sess.run(s)
        feed_dict ={w1:s,w3:1.0}
        fe = sess.run(w2,feed_dict)
        result = tf.math.argmax(fe, 1)
        results.append(sess.run(result)[0])
    shutil.rmtree('./cutImage')
    os.mkdir('./cutImage')
    return results
if __name__ == '__main__':
    run (r"F:\资料下载\123\_763a_0.png")