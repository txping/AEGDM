# AEGDM
This repository contains code to reproduce the experiments in "AEGDM: Adaptive gradient descent with energy and momentum".

## Usage
The aegdm.py file provides a PyTorch implementation of AEGDM,

```python3
optimizer = aegdm.AEGDM(model.parameters(), lr=0.02)
```

## Examples on CIFAR-10 and CIFAR-100
We test AEGDM on the standard CIFAR-10 and CIFAR-100 image classification tasks, comparing with SGD with momentum (SGDM), Adam and AEGD. 
The implementation is highly based on [this repository](https://github.com/Luolc/AdaBound). We also provide a [notebook](./visualization.ipynb) to present our results for this example.

Supported models for CIFAR-10 are VGG, ResNet, DenseNet and CifarNet, for CIFAR-100 are SqueezeNet and GoogleNet.
For VGG, the weight decay is set as '5e-4'; for other architectures, the weight decay is set as '1e-4'.
For DenseNet, the batch size is set as 64; for other architectures, the batch size is set as 128. The initial set of learning rate for each optimizer are:

* SGDM: {0.03, 0.05, 0.1, 0.2, 0.3}
* Adam: {0.0001, 0.0003, 0.0005, 0.001, 0.002}
* AEGD: {0.1, 0.2, 0.3, 0.4}
* AEGDM: {0.005, 0.008, 0.01, 0.02, 0.03}

The best base step size for each method in a certain task can be found in 'curve/pretrained' fold to ease your reproduction.

Followings are examples to train ResNet-32 on CIFAR-10 using AEGDM with a learning rate of 0.008

```bash
python cifar.py --dataset cifar10 --model resnet32 --optim AEGDM --lr 0.008
```
and train SqueezeNet on CIFAR-100 using AEGDM with a learning rate of 0.02
```bash
python cifar.py --dataset cifar100 --model squeezenet --optim AEGDM --lr 0.02
```
The checkpoints will be saved in the `checkpoint` folder and the data points of the learning curve will be saved in the `curve` folder.


## License
[BSD-3-Clause](./LICENSE)
