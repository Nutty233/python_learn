import torchvision
# 全局取消证书验证
# 否则创建数据集时会出现urllib.error.URLError
import ssl

from torch.utils.tensorboard import SummaryWriter

ssl._create_default_https_context = ssl._create_unverified_context


dataset_transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize(100),
    torchvision.transforms.ToTensor()
])

# 调用CIFAR10数据集：存储位置，'True'为训练集'False'为测试集，自定义变换，需提前声明变换操作，下载到本地
# 若本地下载文件与程序访问下载默认文件名相同，则download不用改为false，也不会重新进行下载
# 下载路径可运行程序下载并拷贝；亦可Ctrl+点击数据集名称并通过"url"找到对应地址
train_set = torchvision.datasets.CIFAR10(root="./PytorchDataset",
                                         train=True, transform=dataset_transform, download=True)
test_set = torchvision.datasets.CIFAR10(root="./PytorchDataset",
                                        train=False, transform=dataset_transform, download=True)

writer = SummaryWriter("CIFAR10_Logs")
for i in range(10):
    img, target = test_set[i]
    writer.add_image("test_set", img, i)

writer.close()
print(test_set[0])


# print(test_set[0])
# print(test_set.classes)
#
# img, target = test_set[0]
# print(img)
# print(target)
# print(test_set.classes[target])
#
# img.show()
