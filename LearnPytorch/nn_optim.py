import torch.optim
import torchvision
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sequential
from torch.utils.data import DataLoader


dataset = torchvision.datasets.CIFAR10("PytorchDataset", train=False,
                                       transform=torchvision.transforms.ToTensor())
dataloader = DataLoader(dataset, batch_size=1, shuffle=True, drop_last=True)


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


loss = nn.CrossEntropyLoss()
nutty = Nutty()
# 设置优化器，这里采用随机梯度下降优化器
# lr过大模型训练不稳定，lr过小模型训练慢，一般推荐先用大速率往小调整
optim = torch.optim.SGD(nutty.parameters(), lr=0.01)

for epoch in range(20):
    running_loss = 0.0
    for data in dataloader:  # 这里的循环是遍历了一次dataloader里的数据，需要多轮学习
        imgs, targets = data
        outputs = nutty(imgs)
        # 分类问题用交叉熵计算损失函数
        result_loss = loss(outputs, targets)
        optim.zero_grad()   # 梯度调零，防止干扰
        result_loss.backward()  # 反向传播，求出节点梯度
        optim.step()  # 优化器参数调优
        running_loss += result_loss
        # print(result_loss)
    print(running_loss)

