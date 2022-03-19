# 使用TensorBoader可以直观地进行参数的调试
# SummaryWriter可以实现快速统计功能
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from PIL import Image
# 创建实例,设定存储文件路径
writer = SummaryWriter("logs")
image_path = "data/train/bees_image/17209602_fe5a5a746f.jpg"
img_PIL = Image.open(image_path)
img_array = np.array(img_PIL) # 将PIL读取的图像转换为numpy.ndarray
print(type(img_array))
print(img_array.shape)

# writer.add_image()  # 添加图像来统计
# 添加图像来直观观察：
    # 在训练过程中为model提供了哪些数据
    # 测试model时的输出结果等各种显示
# writer.add_image使用例
    # 从PIL转换到numpy需要在add_image()中用dataformats='HWC'
    # 指定shape中每一个数字/维表示的含义
    # 图像统计结果会在log中创建事件文件
    # 在cmd或Terminal中输入tensorboard --logdir=logs打开
writer.add_image("test", img_array, 4, dataformats='HWC')
    # 这里的global_step仅会影响图像在logs中显示的step编号
    # 可以不更改标题以达到在同一logs窗口下滑动检查图片的效果，更改标题会单独显示

# writer.add_scalar() # 添加标量数据来统计
# writer.add_scalar使用例
    # 绘制y = 2x图像，会在log中创建事件文件
    # 在cmd或Terminal中输入tensorboard --logdir=logs打开
    # 可使用tensorboard --logdir=logs --port=6007(随意)来指定端口，以防和服务器上的其他用户访问冲突
for i in range(100):
    writer.add_scalar(tag="y=2x", scalar_value=2 * i, global_step=i)
    # 等效输入：writer.add_scalar("y=x", 2*i, i)
    # 每改变一次内核，一定要改标题或删除原log文件

writer.close()  # 关闭write
