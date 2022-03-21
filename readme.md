==注：未做数据增广，test正确率很低==



本仓库为在PyTorch中，Fashion-MNIST集，训练ResNet-18

训练模型保存在`_static/model/all_ResNet.pt`中

转化为ONNX模型保存在`_static/model/all_ResNet.onnx`中

ONNX再转换为TensorFlow模型保存在`_static/model/all_ResNet.pb`中



## 环境要求

> 3.6 <= python >=3.8
>
> d2lzh_pytorch
>
> TensorFlow >= 2.8
>
> PyTorch >= 1.0
>
> onnx
>
> onnx_tf
>
> numpy
>
> matplotlib
>
> scikit-image



## 运行说明

### PyTorch->ONNX

直接运行`Resnet2onnx.py`文件

【当删掉`_static/model`文件夹下的模型时，将会重新训练ResNet】

调用顺序：`Resnet2onnx.py => ResNet.py => testONNK.py`

### ONNX->Tensorflow

直接运行`runTensorlow.py`文件

### SuperResolutionNet

这个是官方示例

运行`SRN2onnx.py`文件

调用顺序：`SRN2onnx.py => SuperResolutionNet.py => testONNK.py`



## 具体作用

### 代码

`SuperResolutionNet.py`：SuperResolutionNet模型和它的类，参数为直接下载

`ResNet.py`：ResNet模型及使用Fashion-MNIST训练

`SRN2onnx.py`：SuperResolutionNet从pt转为onnx

`Resnet2onnx.py`：ResNet从pt转为onnx

`getImage`：读取图像，进行YCbCr、大小、维度等转换

`testONNX`：跑了一下SuperResolutionNet和ResNet的onnx模型

`runTensorlow.py`：ONNX存储为pb文件并简单测试

### 模型

在`_static/model`文件夹下，pt、onnx、pb文件

`ResNet.pt`是没有保存训练参数的模型

`all_ResNet.pt`是训练后整个保存

### 图像

在`_static/image`文件夹下

train0-9为我简易测试保存的图像，尺寸为224×224×3

cat.jpg和cat_superres_with_ort.jpg分别为被SuperResolutionNet处理前后图像



## 运行截图

### PyTorch ResNet-18 训练结果

<img src="https://s2.loli.net/2022/03/22/1B7d5vPQyRc89DJ.png" width=80%/>

### ONNX 运行测试

<img src="https://s2.loli.net/2022/03/22/9A3pg4GrUJtZLHK.png" width=80%/>

### TensorFlow 运行测试

<img src="https://s2.loli.net/2022/03/22/GO3Hkjgxno6fz4U.png" width=80%/>

**~~(对不起俺知道这都能识别错很离谱，俺会改的)~~**

