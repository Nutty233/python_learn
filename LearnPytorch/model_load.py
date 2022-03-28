import torch
import torchvision
from torch import nn

# # 加载方式 1（对应保存方式 1）
# model = torch.load("vgg16_method1.pth")
# print(model)

# # 加载方式 2（对应保存方式 2）
# # 新建网络模型结构
# vgg16 = torchvision.models.vgg16(pretrained=False)
# # 加载网络参数
# model_para = torch.load("vgg16_method2.pth")
# # 还原网络
# vgg16.load_state_dict(model_para)
# # model_para = torch.load("vgg16_method2.pth")
# # print(model_para)
# print(vgg16)


# 陷阱
# # 如果只有以下两行会报错：无法访问模型的类
# model = torch.load("nutty_method.pth")
# print(model)

# 陷阱解决方法1：把定义的模型结构复制过来（但不需要初始化：nutty = Nutty()）
class Nutty(nn.Module):
    def __init__(self) -> None:
        super(Nutty, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3)

    def forward(self, x):
        return self.conv1(x)

model = torch.load("nutty_method.pth")
print(model)

# 陷阱解决方法2：
# from model_save import
