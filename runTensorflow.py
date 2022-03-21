#!/usr/bin/env python 
# -*- coding:utf-8 -*-

import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image
import numpy as np
import onnx
import os
from onnx_tf.backend import prepare
from getImage import get_mnist


classes = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
           "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

def onnx2Tensorflow():
    onnx_model = onnx.load("_static/model/all_ResNet.onnx")
    # prepare tensorflow representation
    tf_rep = prepare(onnx_model)
    tf_rep.export_graph("_static/model/all_ResNet.pb")

def test(net):
    img = []
    for i in range(10):
        # 读取图片
        name = "./_static/img/train" + str(i) + ".png"
        if len(img) == 0:
            img = get_mnist(name)
        else:
            img = np.concatenate((img, get_mnist(name)), axis=0)
    print("img", img.shape)
    output = net(input=img)['output']
    print(output.shape)

    # 绘图：5×2，无坐标轴
    plt.figure(10)
    plt.axis('off')
    plt.xticks([])
    plt.yticks([])
    font = dict(fontsize=12, color='r', family='monospace', weight='bold')

    for i in range(10):
        plt.subplot(2, 5, i + 1)
        plt.axis('off')
        plt.xticks([])
        plt.yticks([])
        name = "./_static/img/train" + str(i) + ".png"
        plt.imshow(Image.open(os.path.join(name)))

        # pytorch模型预测
        pred = np.argmax(output[i])
        print("pytorch", classes[pred])
        plt.text(x=0, y=270, s=classes[pred], fontdict=font)

    plt.show()

if __name__ == '__main__':
    print("tensorflow", tf.__version__)
    onnx2Tensorflow()
    net = tf.saved_model.load("_static/model/all_ResNet.pb")
    test(net)
