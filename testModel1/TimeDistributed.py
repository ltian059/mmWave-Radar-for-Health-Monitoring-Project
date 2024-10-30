import torch
import torch.nn as nn

class TimeDistributed(nn.Module):
    def __init__(self, module):
        super(TimeDistributed, self).__init__()
        self.module = module

    def __multi_time(self,size):
        # 调整输入的尺寸。尺寸参数size，将其并进行调整，将样本数和时间步数相乘后放在列表的第一个位置，并保留其他维度不变。最后，返回调整后的尺寸作为元组。
        size_temp = list(size)# 转换为列表类型
        # 72 200 40 3
        size_temp = [size_temp[0]*size_temp[1]]+size_temp[2:]
        # [14400, 40, 3]
        return tuple(size_temp)

    def __dist_time(self,size,batch,time_dim):
        # 用于调整尺寸的辅助方法。它接收一个尺寸参数size、批次数batch和时间步数time_dim。类似于上一个方法，它将尺寸参数转换为列表类型，并在列表的前两个位置插入批次数和时间步数。然后，将其他维度保持不变，并返回调整后的尺寸作为元组。
        size_temp = list(size)
        # 14400 1024
        size_temp = [batch,time_dim]+size_temp[1:]
        # 将变量batch和time_dim作为新列表的前两个元素，然后将原始列表size_temp从索引为1的位置开始的所有元素切片，并将其与前两个元素进行拼接。
        # 72 200 1024
        return tuple(size_temp)

    def forward(self, x):
        # Squash samples and timesteps into a single axis
        # 输入x首先通过调用__multi_time方法进行尺寸调整，将样本数和时间步数合并到一个维度上。然后，调用保存的模块self.module对调整后的输入进行处理，得到输出y。最后，通过调用__dist_time方法重新调整输出的尺寸，将其恢复为样本数、时间步数和输出维度。
        # 保证存储空间连续性
        x_reshape = x.contiguous().view(self.__multi_time(x.size()))  # (samples * timesteps, input_size)

        # 按照现在的格式设置y
        y = self.module(x_reshape)

        y = y.contiguous().view(self.__dist_time(y.size(),x.size(0),x.size(1)))  # (samples, timesteps, output_size)

        return y




