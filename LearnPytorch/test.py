import torch
from PIL import Image
import torchvision
from torch import nn
from torch.nn import *


# 加载测试图像
image_path = "images/dog.png"
image = Image.open(image_path)
# image = image.convert('RGB')
# print(image)


# 定义模型
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


# 加载模型参数（由于之前训练模型用的是GPU，会导致在cpu上运行时报错，故需将模型映射到CPU上）
model = torch.load("model_save/nutty_8.pth", map_location=torch.device('cpu'))
print(model)

"""     另外两种方法
1.  image = image.cuda()
2.  device = torch.device("cpu")
    model.to(device)
"""

# 将测试数据调整为模型输入大小及格式
transfrom = torchvision.transforms.Compose([torchvision.transforms.Resize((32, 32)),
                                            torchvision.transforms.ToTensor()])
image = transfrom(image)
# 输入单张图片时，由于网络需要batch_size，所以需要特别定义以迎合模型尺度
image = torch.reshape(image, (1, 3, 32, 32))
print(image.shape)


# 将模型转换为测试类型，不写也没什么关系，但是要养成良好的代码习惯
model.eval()
with torch.no_grad():  # 节约内存及运算性能
    # 输入模型
    output = model(image)
print(output)
# 将分类结果转换为一种利于解读的方式
print(output.argmax(1))
