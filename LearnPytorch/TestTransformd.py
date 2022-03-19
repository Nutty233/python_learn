# transform主要用于特定格式图像的变换
# 常用工具有：
    # totensor  转换图像格式
    # resize    尺寸变换
# 通过transform.ToTensor解决两个问题：
# 1.transforms该如何使用
# 2.tensor的数据类型
from PIL import Image
# PIL是python内置文件读取方式，读取结果为PIL类型的图像
# OpenCV读取结果为numpy.ndarry，两者都可直接输入到ToTensor中进行转换
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

# 绝对路径：D:\PythonCode\LearnPytorch\data\train\ants_image\0013035.jpg
# 相对路径：data/train/ants_image/0013035.jpg
# 一般win系统利用相对路径，防止绝对路径中转义符“\”产生影响
img_path = "data/train/ants_image/0013035.jpg"
img = Image.open(img_path)
# print(img)

tensor_trans = transforms.ToTensor()    # 调用工具(从class中创建)
tensor_img = tensor_trans(img)  # 使用数据
# print(tensor_img)
# 在tensor类型的图像将包含以下信息：
# device：使用的设备
# requires_grad：···
# (Protected Attributes中)
# _backward_hooks 后项：根据结果反向传播指导调整参数
# _grad：梯度      _grad_fn：梯度方法

# 用tensorboard直观展示图片
write = SummaryWriter('logs')
write.add_image("Tensor_img",tensor_img)

write.close()

