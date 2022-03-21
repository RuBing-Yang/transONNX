#!/usr/bin/env python 
# -*- coding:utf-8 -*-

import torch
import os
import numpy as np
import onnx
import onnxruntime
from PIL import Image
import matplotlib.pyplot as plt
from getImage import get_cat
from getImage import get_mnist
from ResNet import file_to_ResNet
import d2lzh_pytorch as d2l
from skimage import io

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

classes = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
           "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

def to_numpy(tensor):
     if tensor.requires_grad:
         return tensor.detach().cpu().numpy()
     else:
         return tensor.cpu().numpy()


def test_SRNet(x, torch_out):
    onnx_model = onnx.load("_static/model/SuperResolutionNet.onnx")
    onnx.checker.check_model(onnx_model)
    onnx_session = onnxruntime.InferenceSession("_static/model/SuperResolutionNet.onnx")
    onnx_inputs = {onnx_session.get_inputs()[0].name: to_numpy(x)}
    onnx_outs = onnx_session.run(None, onnx_inputs)

    np.testing.assert_allclose(to_numpy(torch_out), onnx_outs[0], rtol=1e-03, atol=1e-05)
    print("Exponnxed model has been tested with ONNXRuntime, and the result looks good!")

    img_y, img_cb, img_cr = get_cat()
    onnx_inputs = {onnx_session.get_inputs()[0].name: to_numpy(img_y)}
    onnx_outs = onnx_session.run(None, onnx_inputs)
    img_out_y = onnx_outs[0]
    img_out_y = Image.fromarray(np.uint8((img_out_y[0] * 255.0).clip(0, 255)[0]), mode='L')
    final_img = Image.merge(
        "YCbCr", [
            img_out_y,
            img_cb.resize(img_out_y.size, Image.BICUBIC),
            img_cr.resize(img_out_y.size, Image.BICUBIC),
        ]).convert("RGB")
    final_img.save("./_static/img/cat_superres_with_ort.jpg")

def test_ResNet(torch_net):
    # 核对转换正确性 assert_allclose
    x = torch.randn(10, 1, 224, 224, requires_grad=True)
    x = x.to(device)
    torch_out = torch_net(x).detach().cpu().numpy()
    print("torch_out shape", torch_out.shape)

    onnx_model = onnx.load("_static/model/all_ResNet.onnx")
    onnx.checker.check_model(onnx_model)
    onnx_session = onnxruntime.InferenceSession("_static/model/all_ResNet.onnx")
    onnx_inputs = {onnx_session.get_inputs()[0].name: to_numpy(x)}
    onnx_outs = onnx_session.run(None, onnx_inputs)
    print("onnx_outs shape", len(onnx_outs), onnx_outs[0].shape)

    np.testing.assert_allclose(torch_out, onnx_outs[0], rtol=1e-03, atol=1e-05)
    print("Exported model has been tested with ONNXRuntime, and the result looks good!")

    img = []
    for i in range(10):
        # 读取图片
        name = "./_static/img/train" + str(i) + ".png"
        if len(img) == 0:
            img = get_mnist(name)
        else:
            img = np.concatenate((img, get_mnist(name)), axis=0)
    print("img", img.shape)
    img_tensor = torch.from_numpy(img)
    onnx_inputs = {onnx_session.get_inputs()[0].name: img}
    print("onnx_inputs", len(onnx_inputs))
    onnx_outs = onnx_session.run(None, onnx_inputs)
    img_tensor = img_tensor.to(device)
    torch_out = torch_net(img_tensor).detach().cpu().numpy()
    print("torch_out", torch_out.shape)
    print("torch_out", torch_out)
    print("onnx_outs", len(onnx_outs), onnx_outs[0].shape)

    # 绘图：5×2，无坐标轴
    plt.figure(10)
    plt.axis('off')
    plt.xticks([])
    plt.yticks([])
    font = dict(fontsize=12, color='r', family='monospace', weight='bold')

    for i in range(10):
        plt.subplot(2,5,i+1)
        plt.axis('off')
        plt.xticks([])
        plt.yticks([])
        name = "./_static/img/train" + str(i) + ".png"
        plt.imshow(Image.open(os.path.join(name)))

        # pytorch模型预测
        pred = np.argmax(torch_out[i])
        print("pytorch", classes[pred])
        plt.text(x=0, y=270, s="[pytorch] " + classes[pred], fontdict=font)

        # onnx模型预测
        pred = np.argmax(np.array(onnx_outs[0][i]))
        print("onnx", classes[pred])
        plt.text(x=0, y=250, s="[onnx] " + classes[pred], fontdict=font)

    plt.show()

if __name__ == '__main__':
    torch_net = file_to_ResNet()
    test_ResNet(torch_net)

    """
    0 T-shirt/top T恤/上衣
    1 Trouser 裤子
    2 Pullover 套衫
    3 Dress 连衣裙
    4 Coat 外套
    5 Sandal 凉鞋
    6 Shirt 衬衫
    7 Sneaker 运动鞋
    8 Bag 包
    9 Ankle boot 踝靴
    """