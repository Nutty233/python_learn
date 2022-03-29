import torch
from torch import nn
from torch.nn import *


# 搭建神经网络（以CIFAR10 model为例）
class Nutty(nn.Module):
    def __init__(self) -> None:
        super(Nutty, self).__init__ ()
        self.model = Sequential(
            Conv2d(in_channels=3, out_channels=32, kernel_size=5, padding=2),
            MaxPool2d(kernel_size=2),
            Conv2d(in_channels=32, out_channels=32, kernel_size=5, padding=2),
            MaxPool2d(kernel_size=2),
            Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2),
            MaxPool2d(kernel_size=2),
            Flatten(),
            Linear(1024, 64),  # 也可以是64*4*4
            Linear(64, 10)
        )

    def forward(self, x):
        return self.model(x)


# 测试网络正确性
if __name__ == '__main__':
    nutty = Nutty()
    # 一般的验证思路是输入一个确定尺寸，看输出尺寸是否符合预期
    input = torch.ones(64, 3, 32, 32)
    output = nutty(input)
    print(output.shape)