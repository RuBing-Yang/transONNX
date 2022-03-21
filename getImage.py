#!/usr/bin/env python 
# -*- coding:utf-8 -*-
from PIL import Image
import torchvision.transforms as transforms
from skimage import io
import numpy as np

def get_cat():
    img = Image.open("./_static/img/cat.jpg")
    resize = transforms.Resize([224, 224])
    img = resize(img)

    img_ycbcr = img.convert('YCbCr')
    img_y, img_cb, img_cr = img_ycbcr.split()

    to_tensor = transforms.ToTensor()
    img_y = to_tensor(img_y)
    img_y.unsqueeze_(0)
    return img_y, img_cb, img_cr

def get_mnist(name):
    img = np.float32(io.imread(name))
    img = img[:,:,0] / 255.0
    img = np.float32(img).reshape((1, 1, img.shape[0], img.shape[1]))
    return img

