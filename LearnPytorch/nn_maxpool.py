import torch
import torchvision
from torch import nn
from torch.nn import MaxPool2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10("PytorchDataset", train=False, transform=torchvision.transforms.ToTensor(),
                                       download=True)
# 加载数据集
dataloader = DataLoader(dataset, batch_size=64)


# input = torch.tensor([[1, 2, 0, 3, 1],
#                       [0, 1, 2, 3, 1],
#                       [1, 2, 1, 0, 0],
#                       [5, 2, 3, 1, 1],
#                       [2, 1, 0, 1, 1]], dtype=torch.float32)  # 池化操作一般不能对整型进行操作
# input = torch.reshape(input, (-1, 1, 5, 5))
# print(input.shape)


class Nutty(nn.Module):
    def __init__(self) -> None:
        super(Nutty, self).__init__()
        self.maxpool1 = MaxPool2d(kernel_size=3, ceil_mode=False)

    def forward(self, input):
        output = self.maxpool1(input)
        return output

nutty = Nutty()
# output = nutty(input)
# print(output)

writer = SummaryWriter("logs/nn_maxpool_logs")
step = 0
# 将每张图像放入神经网络卷积
for data in dataloader:
    imgs, targets = data
    output = nutty(imgs)
    writer.add_images("input", imgs, step)
    writer.add_images("output", output, step)
    step += 1

writer.close()