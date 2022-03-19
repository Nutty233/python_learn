# transform主要用于特定格式图像的变换
# 常用工具有：
    # totensor  转换图像格式
    # resize    尺寸变换
# 通过transform.ToTensor解决两个问题：
# 1.transforms该如何使用
# 2.tensor的数据类型
from PIL import Image
from torchvision import transforms

# 绝对路径：D:\PythonCode\LearnPytorch\data\train\ants_image\0013035.jpg
# 相对路径：data/train/ants_image/0013035.jpg
# 一般win系统利用相对路径，防止绝对路径中转义符“\”产生影响
img_path = "data/train/ants_image/0013035.jpg"
img = Image.open(img_path)
print(img)
tensor_trans = transforms.ToTensor()    # 调用工具(从class中创建)
tensor_img = tensor_trans(img)  # 使用数据

print(tensor_img)
