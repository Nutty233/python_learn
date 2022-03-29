import torch
import torchvision
from torch import nn
from torch.nn import *
from torch.utils.data import DataLoader
from torch.utils.tensorboard import *
import time


# 定义训练设备
device = torch.device("cuda")
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 利用语法糖简写
# 准备训练数据集
train_data = torchvision.datasets.CIFAR10(root="PytorchDataset", train=True,
                                          transform=torchvision.transforms.ToTensor(), download="True")
# 准备测试数据集
test_data = torchvision.datasets.CIFAR10(root="PytorchDataset", train=False,
                                         transform=torchvision.transforms.ToTensor(), download="True")
# 查看数据集长度
train_data_size = len(train_data)
test_data_size = len(test_data)
print("训练数据集的长度为：{}".format(train_data_size))  # format字符串格式化
# 选中之后ctrl + d = ctrl + c 之后 ctrl + v
print("测试数据集的长度为：{}".format(test_data_size))  # format字符串格式化

# 利用dataloader加载数据集
train_dataloader = DataLoader(train_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)


# 创建网络模型
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


nutty = Nutty()
# 调用训练设备
nutty.to(device)

# 定义损失函数
loss_fn = CrossEntropyLoss()
loss_fn = loss_fn.to(device)

# 定义优化器
learning_rate = 0.01  # 或用科学计数法1e-2
optimizer = torch.optim.SGD(params=nutty.parameters(), lr=learning_rate)

# 设置训练网络的参数
# 记录训练的次数
total_train_step = 0
# 记录测试的次数
total_test_step = 0
# 设置训练的轮数
epoch = 10

# 添加tensorboard显示损失函数变化（或许也可以用Matplotlib绘制动态曲线图？？？）
writer = SummaryWriter("logs/train_logs")
# 记录开始时间
start_time = time.time()
for i in range(epoch):
    print("------第 {: >2d} 轮训练开始------".format(i+1))

    # 训练步骤开始
    nutty.train()  # 将网络设置训练模式，网络模型中有Dropout，BatchNorm等层时有作用，需调用，平时就算没有特定层写上也无妨
    for data in train_dataloader:
        imgs, targets = data
        imgs = imgs.to(device)
        targets = targets.to(device)
        outputs = nutty(imgs)
        # 损失函数计算损失
        loss = loss_fn(outputs, targets)

        # 优化器优化模型
        # 将梯度清零
        optimizer.zero_grad()
        # 损失函数的反向传播
        loss.backward()
        # 进行优化
        optimizer.step()

        total_train_step += 1
        if total_train_step % 100 == 0:
            # 记录100次训练后的结束时间
            end_time = time.time()
            print(end_time - start_time)
            print("训练次数：{}，Loss：{}".format(total_train_step, loss.item()))   # tensor数据.item()：把tensor数据转化为int
            writer.add_scalar("train_loss", loss.item(), total_train_step)

    # 每轮训练之后运用测试集进行一轮测试，以验证模型效果
    # 以模型在测试集上的损失（正确率）来评估模型
    # 测试步骤开始
    nutty.eval()  # 将网络设置测试模式，网络模型中有Dropout，BatchNorm等层时有作用，需调用，平时就算没有特定层写上也无妨
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad(): # 在with下的代码不设置梯度，以禁止在测试时调优
        for data in test_dataloader:
            imgs, targets = data
            imgs = imgs.to(device)
            targets = targets.to(device)
            outputs = nutty(imgs)
            loss = loss_fn(outputs, targets)
            total_test_loss += loss.item()
            # 计算整体正确率
            accuracy = (outputs.argmax(1)==targets).sum()
            total_accuracy += accuracy

    print("整体测试集上的Loss:{}".format(total_test_loss))
    print("整体测试集上的正确率:{}".format(total_accuracy/test_data_size))
    writer.add_scalar("test_loss", total_test_loss, total_test_step)
    writer.add_scalar("test_accuracy", total_accuracy/test_data_size, total_test_step)
    total_test_loss += 1

    # 保存每一轮训练后的模型
    torch.save(nutty, "model_save/nutty_{}.pth".format(i))
    # [官方推荐方式] torch.save(nutty.state_dict(), "model_save/nutty_{}.pth".format(i))
    print("模型已保存,存储路径为：model_save/nutty_{}.pth".format(i))

writer.close()
