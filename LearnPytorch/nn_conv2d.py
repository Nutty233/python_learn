import torch
import torchvision
from torch import nn
from torch.nn import Conv2d
from torch.utils.data import DataLoader

# 加载数据集
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10("PytorchDataset", train=False, transform=torchvision.transforms.ToTensor(),
                                       download=True)
# 加载数据集
dataloader = DataLoader(dataset, batch_size=64)


# 搭建神经网络
class Nutty(nn.Module):
    def __init__(self) -> None:  # 初始化方法
        super(Nutty, self).__init__()  # 父类初始化，以便子类里访问父类的同名属性，而又不想直接引用父类的名字
        self.conv1 = Conv2d(in_channels=3, out_channels=6, kernel_size=3,
                            stride=1, padding=0)  # 定义卷积层,卷积核为默认，边缘补充若默认为0的话会导致输出图像尺寸缩小

    def forward(self, x):
        x = self.conv1(x)  # 将x放入卷积层中
        return x


# 初始化网络
nutty = Nutty()
print(nutty)

writer = SummaryWriter("logs/nn_conv2d_logs")
step = 0
# 将每张图像放入神经网络卷积
for data in dataloader:
    imgs, targets = data
    output = nutty(imgs)
    # print(imgs.shape)
    # print(output.shape)
    # 输入的大小：torch.Size([64, 3, 32, 32])
    writer.add_images("input", imgs, step)
    # 输出的大小：torch.Size([64, 6, 30, 30])
    # 由于输出channel数与输入不同，会导致board无法显示，可显示shape：torch.Size([64, 3, 30, 30])
    # 可取一个不是很严谨的方法解决此问题：对输出进行reshape（强行切开）
    output = torch.reshape(output, (-1, 3, 30, 30))  # 不知道会变化成什么样的参数可赋值-1，程序会自动匹配
    writer.add_images("output", output, step)
    """ 也可取一个严谨的方法：通道分离(完善)
        input_channel = imgs.shape[1]
        output_channel = output.shape[1]
        group_num = output_channel/input_channel
        for i in range(group_num):
            ...
    """
    step += 1

writer.close()

