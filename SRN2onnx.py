#!/usr/bin/env python 
# -*- coding:utf-8 -*-
# Some standard imports
import io
from torch import nn
import torch.utils.model_zoo as model_zoo
import torch.onnx

from SuperResolutionNet import SuperResolutionNet
from testONNX import test_SRNet

if __name__ == '__main__':
    # Create the super-resolution model by using the above model definition.
    torch_model = SuperResolutionNet(upscale_factor=3)

    # Load pretrained model weights
    model_url = 'https://s3.amazonaws.com/pytorch/test_data/export/superres_epoch100-44c6958e.pth'
    batch_size = 1  # just a random number

    # Initialize model with the pretrained weights
    map_location = lambda storage, loc: storage
    if torch.cuda.is_available():
        map_location = None
    torch_model.load_state_dict(model_zoo.load_url(model_url, map_location=map_location))

    # set the model to inference mode
    torch_model.eval()

    # Input to the model
    x = torch.randn(batch_size, 1, 224, 224, requires_grad=True)
    torch_out = torch_model(x)

    # Export the model
    torch.onnx.export(torch_model,  # model being run
                      x,  # model input (or a tuple for multiple inputs)
                      "_static/model/SuperResolutionNet.onnx",  # where to save the model (can be a file or file-like object)
                      export_params=True,  # store the trained parameter weights inside the model file
                      opset_version=10,  # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names=['input'],  # the model's input names
                      output_names=['output'],  # the model's output names
                      dynamic_axes={'input': {0: 'batch_size'},  # variable length axes
                                    'output': {0: 'batch_size'}})

    test_SRNet(x, torch_out)
