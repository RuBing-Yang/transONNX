#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import os
import torch.utils.model_zoo as model_zoo
import torch.onnx
import torch.nn.functional as F

from ResNet import train_fashion_mnist_Resnet
from ResNet import file_to_ResNet
from testONNX import test_ResNet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':
    print("pytorch版本:", torch.__version__)

    # 没训练过 则需要pytorch从头训练resnet
    if os.path.exists('_static/model/all_ResNet.pt'):
        torch_net = file_to_ResNet()
    else:
        torch_net = train_fashion_mnist_Resnet()

    x = torch.randn(10, 1, 224, 224, requires_grad=True)
    x = x.to(device)
    torch_out = torch_net(x)
    # print(net)
    print("x shape", x.shape)

    torch.onnx.export(torch_net,
                      x,
                      "_static/model/all_ResNet.onnx",
                      export_params=True,  # store the trained parameter weights
                      opset_version=11,
                      do_constant_folding=False,  # constant folding for optimization
                      input_names=['input'],
                      output_names=['output'])

    # 因为报错删去下面两个参数
    # Error: Failed to export an ONNX attribute 'onnx::Gather',
    # since it's not constant, please try to make things
    # (e.g., kernel size) static if possible

                      # dynamic_axes={'input': {0: 'batch_size'},  # variable length axes
                      #               'output': {0: 'batch_size'}})

    # 测试
    test_ResNet(torch_net)