import torch
import torchvision
from torch import nn
from torch.nn import ReLU, Sigmoid
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

input = torch.tensor([[1, -0.5],
                      [-1, 3]])
input = torch.reshape(input, (-1, 1, 2, 2))
print(input.shape)

dataset = torchvision.datasets.CIFAR10("PytorchDataset", train=False,download=True,
                                       transform=torchvision.transforms.ToTensor())
dataloader = DataLoader(dataset,batch_size=64)

class Nutty(nn.Module):
    def __init__(self) -> None:
        super(Nutty, self).__init__()
        self.relu1 = ReLU()  # inplace:替换；若为True直接将结果替换input的只，若为False保留input原始值并输出新的结果，防止数据丢失
        self.sigmoid1 = Sigmoid()

    def forward(self, input):
        output = self.sigmoid1(input)
        return output

nutty = Nutty()
writer = SummaryWriter("logs/nn_relu_logs")
step = 0
# output = nutty(input)
# print(output)
for data in dataloader:
    imgs, targets = data
    writer.add_images("inout", imgs, step)
    output = nutty(imgs)
    writer.add_images("output", output, step)
    step += 1
writer.close()
