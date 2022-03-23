# PyTorch基础学习相关
本文件于2022/03/18创建于本地，于2022/03/19首次上传至github
本文件夹用于存储PyTorch基础学习相关数据集及代码
其中数据集由教程提供，代码部分依照教程手动输入
并依照自己的理解进行修改及备注，与视频教程稍有不同

## 文件夹程序及其作用：
    **main.py: **
        判断Torch安装是否成功

    **ReadData.py:**
        创建以PIL为核心的图像打开类及数据集的调用和合并
        (也可用CV2，但需注意后续转换为numpy格式时的提前转换)

    **RenameDataset.py:**
        用于依照训练集生成相应标注文件(.txt)

    **TestTensorboard.py:**
        利用TenserBoard的SummaryWriter函数（快速拥挤功能）
        生成logs文件以完成标量数据统计分析及直观纵览图像输入情况
        (需要在控制台输入指令在网页中打开)

    **TestTransformd.py:**
        调用TestTransformd中的函数(工具)用于图像的格式及尺寸变换等操作
        通过输出Tensor图像属性理解tensor类型含有神经网络编程所需参数的作用

    **UsefulTransform.py**
        常见transform介绍（把握输入、输出、作用），推荐常看官方文档（比网上经验准确）
        关注方法需要什么参数，有官方默认值参数的一般不需要自己  定义
        类中特殊的实例方法（__call__()等）介绍
    
    **DatasetTransform.py**
        利用transform批处理Pytorch官网提供的开源数据集
    
    **Dataloader.py**
        利用Dataloader创建数据集加载工具，用于为网络提供数据输入
        实际训练中，一般采用程序中相同的策略：利用for循环将imgs传入神经网络作为输入
    
    **nn_module.py**
        神经网络的基本骨架——nn.Module的使用(利用Torch.nn中的containers进行神经网络骨架搭建) 
        前向传播：input-->forward-->output
        非线性处理：relu；卷积conv


## 教程及数据集出处：
    B站 PyTorch深度学习快速入门教程（绝对通俗易懂！）【小土堆】
    （B站视频号 74281036）
    感谢该UP主！
