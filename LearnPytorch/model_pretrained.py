import torchvision

# train_data = torchvision.datasets.ImageNet("PytorchDataset", split="train", download=True,
#                                            transform=torchvision.transforms.ToTensor())
from torch import nn

vgg16_false = torchvision.models.vgg16(pretrained=False)    # 若为True，则单单加载模型参数采用默认参数
vgg16_true = torchvision.models.vgg16(pretrained=True)  # 若为True，则会从网上下载训练过的参数
# 训练过的模型bias将不再是0

# print(vgg16_true)
# 可以看出最后VGG16将输出1000个类（out_features=1000），若想针对各数据集（比如只有10个类的CIFAR10），则需要更改原模型输出，或在模型后追加一层更改输出

train_data = torchvision.datasets.CIFAR10("PytorchDataset", train=True,
                                          transform=torchvision.transforms.ToTensor(), download=True)
# 追加
# # 为VGG16追加一层，加在外部
# vgg16_true.add_module('add_linear', nn.Linear(1000, 10))
# 为VGG16追加一层，加在classifier部中
vgg16_true.classifier.add_module('add_linear', nn.Linear(1000, 10))
# print(vgg16_true)


# 修改
vgg16_false.classifier[6] = nn.Linear(4096, 10)
# print(vgg16_false)


# 删除（自己网上查的）
# 方案1：创建空顺序层替换原层（内层或外层）
# vgg16_false.classifier[6] = nn.Sequential()
# print(vgg16_false)

# 方案2：创建标识层并替换原层（内层或外层）
vgg16_false.add_module('add_linear', nn.Linear(1000, 10))
print(vgg16_false)
class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x
vgg16_false.classifier[6] = Identity()
vgg16_false.add_linear = Identity()
print(vgg16_false)