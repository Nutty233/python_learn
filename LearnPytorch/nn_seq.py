import torch
import torchvision
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sequential
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


# class Nutty(nn.Module):
#     def __init__(self) -> None:
#         super(Nutty, self).__init__()
#         self.conv1 = Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=1, padding=2)
#         self.maxpool1 = MaxPool2d(kernel_size=2)
#         self.conv2 = Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=2)
#         self.maxpool2 = MaxPool2d(kernel_size=2)
#         self.conv3 = Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2)
#         self.maxpool3 = MaxPool2d(kernel_size=2)
#         self.flatten1 = Flatten()
# 在复刻模型时须注意输入和输出的数据量，判断是否需要加入线性层以保证输出数据量与下一步的输入数据量相同
# 两个相邻线性层间通常有激活层以防止过拟合,本程序没有
# 快速知道flatten之后的形状方法：输出flatten之后的形状，第二个值即为linear的in_features
#         self.linear1 = Linear(in_features=1024 ,out_features=64)  # 1024=64*4*4 在flatten后有个隐藏的线性层将1024 转化为64
#         self.linear1 = Linear(in_features=64 ,out_features=10)
#
#     def forward(self, input):
#         output = self.conv1(self.maxpool1(self.conv2(self.maxpool2(
#             self.conv3(self.maxpool3(self.flatten1(self.linear1(input))))))))
#         return output


# 简便写法
class Nutty(nn.Module):
    def __init__(self) -> None:
        super(Nutty, self).__init__()
        self.model = Sequential(
            Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=1, padding=2),
            MaxPool2d(kernel_size=2),
            Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=2),
            MaxPool2d(kernel_size=2),
            Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2),
            MaxPool2d(kernel_size=2),
            Flatten(),
            Linear(1024, 64),
            Linear(64, 10)
        )

    def forward(self, input):
        output = self.model(input)
        return output

nutty = Nutty()
print(nutty)

# 为了验证网络正确性，通常需要一个假想输入，若输入不符合逻辑就会报错
input = torch.ones((64, 3, 32, 32))
output = nutty(input)
print(output.shape)

dataset = torchvision.datasets.CIFAR10("PytorchDataset", train=False,
                                       transform=torchvision.transforms.ToTensor())
dataloader = DataLoader(dataset, batch_size=64, shuffle=True, drop_last=True)

writer = SummaryWriter("logs/nn_seq_logs")
# 查看网络结构（流程图），通过双击可查看各部分细节
writer.add_graph(nutty, input)
writer.close()
# steps = 0
#
# for data in dataloader:
#     imgs, targets = data
#     output = nutty(imgs)
#     # writer.add_images("input", imgs, steps)
#     print(output)
#
# writer.close()