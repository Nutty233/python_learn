#PyTorch基础学习相关 
本文件于2022/03/18创建于本地，于2022/03/19首次上传至github 
本文件夹用于存储PyTorch基础学习相关数据集及代码 
其中数据集由教程提供，代码部分依照教程手动输入 
并依照自己的理解进行修改及备注，与视频教程稍有不同 

##文件夹程序包括：
    main.py:
        判断Torch安装是否成功

    ReadData.py:
        创建以PIL为核心的图像打开类及数据集的调用和合并
        (也可用CV2，但需注意后续转换为numpy格式时的提前转换)

    RenameDataset.py:
        用于依照训练集生成相应标注文件(.txt)

    TestTensorboard.py:
        利用TenserBoard的SummaryWriter函数（快速拥挤功能）
        生成logs文件以完成标量数据统计分析及直观纵览图像输入情况

    TestTransformd.py:
        调用TestTransformd中的函数(工具)用于图像的格式及尺寸变换等操作
        通过···理解tensor类型数据的作用

##教程及数据集出处：
    B站 PyTorch深度学习快速入门教程（绝对通俗易懂！）【小土堆】
    （B站视频号 74281036）
    感谢该UP主！
