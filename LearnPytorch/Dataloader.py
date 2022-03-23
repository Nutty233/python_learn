import torchvision

#准备测试集
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

test_data = torchvision.datasets.CIFAR10("./PytorchDataset", train=False,
                                         transform=torchvision.transforms.ToTensor())
# 设置加载器,从test_data依照步长取四个数据，访问完成后打乱，最后步长除不尽的部分舍弃(True)。win系统下如果线程数不为0则有可能出错
test_loader = DataLoader(dataset=test_data, batch_size=64, shuffle=False,
                         num_workers=0, drop_last=True)

# 测试数据集中第一张图片及target
img, target = test_data[0]
print(img.shape)
print(target)

writer =SummaryWriter("dataloader")
step = 0
for data in test_loader:
    imgs, targets = data
    # print(imgs.shape)
    # print(targets)
    writer.add_images("test_data_droplist", imgs, step)
    step = step+1

# 测试shuffle的打乱(True)及同序(False)效果
for epoch in range(2):
    step = 0
    for data in test_loader:
        imgs, targets = data
        writer.add_images("Epoch:{}".format(epoch), imgs, step)
        step = step + 1

writer.close()



