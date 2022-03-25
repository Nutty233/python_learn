import torch
import torchvision
from torch import nn
from torch.nn import Linear
from torch.utils.data import DataLoader

dataset = torchvision.datasets.CIFAR10("PytorchDataset", train=False, transform=torchvision.transforms.ToTensor(),
                                       download=True)

dataloader = DataLoader(dataset, batch_size=64, drop_last=True)  # 由于不同图片尺寸不同，遍历到最后不够尺寸就会报错


class Nutty(nn.Module):
    def __init__(self) -> None:
        super(Nutty, self).__init__()
        self.linear1 = Linear(196608, 10)

    def forward(self, input):
        output = self.linear1(input)
        return output

nutty = Nutty()


for data in dataloader:
    imgs, tagets = data
    print(imgs.shape)
    # output = torch.reshape(imgs, (1, 1, 1, -1))
    output = torch.flatten(imgs)  # 若最终结果只要一位数组（torch.shape中有三个都是1时），可用本行替代上一行，操作更为简单
    print(output.shape)
    output = nutty(output)
    print(output.shape)
