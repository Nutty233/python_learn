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
for data in dataloader:
    imgs, targets = data
    outputs = nutty(imgs)
    # writer.add_images("input", imgs, steps)
    # # 先看一下outputs和targets长什么样，再决定选择什么样的损失函数
    # print(outputs)  # 所有类的结果：分为该类的概率
    # print(targets)  # 真实类名
    # 分类问题用交叉熵计算损失函数
    result_loss = loss(outputs, targets)
    result_loss.backward()  # 反向传播，为神经网络中的各节点各类预测值提供grad，即更新参考。反向传播与forward相对应
    print(result_loss)

