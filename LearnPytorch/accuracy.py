# 正确率：分类问题的一种特有衡量指标
import torch

outputs = torch.tensor([[0.1, 0.2],
                       [0.3, 0.4]])

print(outputs.argmax(1))  # 横向返回最大值所在位置

preds = outputs.argmax(1)
targets = torch.tensor([0, 1])
print((preds == targets).sum())  # 计算对应位置相等的个数
