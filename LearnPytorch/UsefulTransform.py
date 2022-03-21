from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

writer = SummaryWriter("logs")
img = Image.open("images/43l3v6.jpg")
print(img)

# ToTensor
trans_totensor = transforms.ToTensor()
img_tensor = trans_totensor(img)
writer.add_image("ToTensor", img_tensor)   # 输出至TensorBoard


# ToPILImage：将tensor_img或ndarry转换为PIL


# Normalize：归一化
# output[channel] = (input[channel] - mean[channel]) / std[channel]
# 2 * value - 1
print(img_tensor[0][0][0])
trans_norm = transforms.Normalize([0.5, 0.5, 0.5],
                                  [0.5, 0.5, 0.5]) # 均值及标准差数需与通道保持一致
img_norm = trans_norm(img_tensor)
print(img_norm[0][0][0])
writer.add_image("Normalize", img_norm)


# Resize：调整尺寸，输入需要一张PIL图像
# 联想输入时需注意大小写，或"File--Setting--搜索"Code Completion--取消Match case"
print(img.size)
trans_resize = transforms.Resize((512, 512))    # 如果指定长宽需要注意参数为一个序列(字典型)
# img(PIL) --resize--> img_resize(PIL)
img_resize = trans_resize(img)
print(img_resize)
# img_resize(PIL) --totensor--> img_resize(tensor)
img_resize = trans_totensor(img_resize)
writer.add_image("Resize", img_resize, 0)
print(img_resize)


# compose - resize：通过compose创建序列(打包流程，一个语句实现多个流程)进行resize
trans_resize_com = transforms.Resize(512)   # 如果只指定一个值，图像长宽比例将不会改变，并将最短边缩小至参数值
# img(PIL) --compose_resize--compose_totensor--> img_resize_com(tensor)
# 括号中参数形式为"[第一个变换，第二个变换...]"需注意两个变换间的输出与输入类型是否匹配：
trans_compose = transforms.Compose([trans_resize_com, trans_totensor])
img_resize_com = trans_compose(img)    # 需要参照compose中的第一个变换决定输入数据的类型
writer.add_image("Resize", img_resize_com, 1)

"""
不知道数据类型时：
1. print
2. print(type())
3. 加断点,debug
4. 看官方文档
5. 上网查或多试错
"""


# RandomCorp：随机裁剪，随机在PIL图像中裁剪指定长宽(指定序列，用字典型)的区域，或指定边长的正方形区域
trans_random = transforms.RandomCrop(512)
trans_compose_2 = transforms.Compose([trans_random, trans_totensor])
for i in range(10):
    img_crop = trans_compose_2(img)
    writer.add_image("RandomCrop", img_crop, i)


writer.close()
