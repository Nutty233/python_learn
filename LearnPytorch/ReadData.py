# 从torch的“常用区”中的“数据”区导入 Dataset
# 可用help()查看该类中需要自己重写（重新定义）的函数
from torch.utils.data import Dataset
from PIL import Image
import os

# 创建自己的类MyData并继承Dataset类
class MyData(Dataset):
    # 初始化类：根据类创建实例时需要运行的函数，为整个Class提供全局变量
    # 一般将根目录(root_dir)和目标目录分别存放(label_dir)
    # 调用时运用os.path.join(root_dir, label_dir)防止破折号语法错误
    def __init__(self, root_dir, label_dir):
        # 类中变量前加self可将一个函数中的变量传递给其它函数，相当于创建了一个全局变量
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.path = os.path.join(self.root_dir, self.label_dir)
        # 通过索引获取图片地址
        self.img_path = os.listdir(self.path)

    # 用来读取图片
    # 当实例对象做P[key]运算时，就会调用类中的__getitem__()方法
    def __getitem__(self, idx):
        img_name = self.img_path[idx]
        img_item_path = os.path.join(self.root_dir, self.label_dir, img_name)
        img = Image.open(img_item_path)
        label = self.label_dir
        return img, label

    # 用来获取数据长度
    def __len__(self):
        return len(self.img_path)

# 运用以上类创建实例函数
root_dir = "dataset/train"
ants_label_dir = "ants_image"
bees_label_dir = "bees_image"
ants_dataset = MyData(root_dir, ants_label_dir)
bees_dataset = MyData(root_dir, bees_label_dir)

# 合并两个数据集
train_dataset = ants_dataset + bees_dataset

# 获取第一个变量
print(train_dataset[0])

print(len(train_dataset))