import torch
from torch import nn


# 定义神经网络（模板）
class Nutty(nn.Module):
    # 快速重写继承类的方法：
    # Alt+Insert-->Override Methods-->选择想重写的函数
    def __init__(self) -> None:
        super().__init__()

    def forward(self, input):
        output = input + 1
        return output


# 使用模板创建神经网络
nutty = Nutty()
model_input = torch.tensor(1.0)   # 输入tensor格式的数字，直接输入数字则为相应数字格式
model_output = nutty(model_input)

print(model_input)
print(model_output)
