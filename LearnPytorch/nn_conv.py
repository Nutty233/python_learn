import torch
import torch.nn.functional as F


# 定义输入（图像）
input = torch.tensor([[1, 2, 0, 3, 1],
                      [2, 1, 2, 3, 1],
                      [1, 2, 1, 0, 0],
                      [5, 2, 3, 1, 1],
                      [2, 1, 0, 1, 1]])  # 开始有几个中括号就是几维矩阵
# 定义卷积核
kernel = torch.tensor([[1, 2, 1],
                       [0, 1, 0],
                       [2, 1, 0]])
print(input.shape)
print(kernel.shape)
# 由于conv2d的input和weight分别需要4个输入，而当前input和kernel分别只有两个
# 因此需要利用pytorch的工具进行尺寸变换,参数为输入和变换形态（N,C,H,W）
# 变换形态括号中参数分别为N：batch size（输入图片数量）,通道数,高,宽
input = torch.reshape(input, (1, 1, 5, 5))
kernel = torch.reshape(kernel, (1, 1, 3, 3))
# 由于输入的变换形态元组为四维，在reshape后输出数组也已变为四维
print(input)
print(kernel)
print(input.shape)
print(kernel.shape)

# 变为四维的另一个解释：输出维数 = 输入维数 - 卷积核大小/步长 + 1
output = F.conv2d(input, kernel, stride=1)
print(output)

output2 = F.conv2d(input, kernel, stride=2)
print(output2)

output3 = F.conv2d(input, kernel, stride=1, padding=1)
print(output3)
