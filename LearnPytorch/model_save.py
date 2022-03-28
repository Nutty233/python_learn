import torch
import torchvision
from torch import nn

vgg16 = torchvision.models.vgg16(pretrained=False)

# 保存方式 1: 同时保存了网络模型的结构和参数
torch.save(vgg16, "vgg16_method1.pth")

# 保存方式 2: 以字典形式保存网络模型的参数（状态）[官方推荐]
torch.save(vgg16.state_dict(), "vgg16_method2.pth")


# 陷阱
class Nutty(nn.Module):
    def __init__(self) -> None:
        super(Nutty, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3)

    def forward(self, x):
        return self.conv1(x)


nutty = Nutty()
torch.save(nutty, "nutty_method.pth")
