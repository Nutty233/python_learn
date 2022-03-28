import torch
from torch.nn import L1Loss, MSELoss, CrossEntropyLoss

inputs = torch.tensor([1, 2, 3], dtype=torch.float32)
targets = torch.tensor([1, 2, 5], dtype=torch.float32)

inputs = torch.reshape(inputs, (1, 1, 1, 3))  # 一定要注意input和target的形状
targets = torch.reshape(targets, (1, 1, 1, 3))

loss = L1Loss(reduction='sum')  # 默认为求差取平均,可定义reduction参数改为相加等操作
result = loss(inputs, targets)
print(result)

loss_mse = MSELoss()
result_mse = loss_mse(inputs, targets)
print(result_mse)

x = torch.tensor([0.1, 0.2, 0.3])
y = torch.tensor([1])  # 想算的类名称（第“1”类）
x = torch.reshape(x, (1, 3))
loss_cross = CrossEntropyLoss()
result_cross = loss_cross(x, y)
print(result_cross)
