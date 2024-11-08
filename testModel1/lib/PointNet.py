from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F # F是别名


class STN3d(nn.Module):# input transform(T-Net，对原样本进行一定的卷积和全连接操作得到变换矩阵，相当于一个缩小版pointnet)生成3*3矩阵，负责姿态变换，拿点云数据乘这个矩阵，出来的大小不变
    def __init__(self, input_channels = 3):
        super(STN3d, self).__init__()
        self.input_channels = input_channels
        # mlp
        # 上文提到的mlp均由卷积结构完成
        # 比如3维映射到64维，利用64个1*3的卷积核
        # 将一个3通道的输入数据（假设通道顺序为R、G、B）与64个1x3卷积核进行卷积操作，每个卷积核会分别对R、G、B三个通道进行加权和融合，生成一个新的通道特征，这样就可以将输入数据的通道数从3通道变换成64通道。
        # 1维卷积层Conv1d，与二维卷积类似，比如数据维度是8，过滤器维度是5，步长是1，向右移动过滤器，这样得到8-5+1=4维的数据
        # 一些参数：
        # in_channels：输入数据的通道数，对应于输入数据的维度，例如，如果输入数据是3通道的，则 in_channels 为3。
        # out_channels：输出数据的通道数，即卷积层的卷积核数量，可以设置为64，用于将输入数据从3通道转换成64通道。
        # kernel_size：卷积核的大小，可以通过一个整数或者一个元组来指定，例如，可以设置为 kernel_size=3 表示卷积核的大小为3。
        # 其他参数如 stride、padding、dilation 等用于设置卷积操作的步长、填充和膨胀等（填充是指在输入数据的边缘周围添加额外的值（通常为0），使得输入数据的维度在卷积操作前后保持一致；
        # 膨胀是指在卷积操作中，卷积核内部的元素之间添加一些间隔，使得卷积核的有效感受野增大，从而能够更好地捕捉输入数据中的长距离依赖关系。）。
        self.conv1 = torch.nn.Conv1d(self.input_channels, 64, 1)# 输入通道为3，输出通道为64，卷积核大小为1（实际卷积核是1*3）
        self.conv2 = torch.nn.Conv1d(64, 128, 1)# 随机初始化，模型在训练过程中会通过反向传播自动学习合适的卷积核权重
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)# 最大池化是把n个点取最大值变成1*1024（batchsize是1）
        # 三个全连接层，当前层每一个神经元都要连接上一层的所有神经元
        # 假设上一层有 n 个神经元，当前层有 m 个神经元，那么在全连接层中，总共需要 n * m 个权重参数
        # （n 个输入特征对应 n 个权重参数，每个输入特征连接到 m 个神经元）
        # 以及 m 个偏置参数（每个神经元对应一个偏置参数）
        self.fc1 = nn.Linear(1024, 512)# 1024上一层，512当前层
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, self.input_channels * self.input_channels)# 3*k，k是输入数据的维度，这之后会有一个3*3的reshape，第二次姿态调整是64*64

        self.relu = nn.ReLU()# 激活函数是每一个神经元计算加权值后输入到激活函数里，然后给下一个连接层

        # 归一化防过拟合，用于每一个卷积层和全连接层
        # 计算输入数据在每个通道上的均值和标准差。这里的输入数据是一维的
        # 使用计算得到的均值和标准差对输入数据进行归一化，将其转换为零均值和单位方差的数据。
        # 对归一化后的数据进行缩放和平移操作，使用可学习的缩放因子（称为 weight）和平移因子（称为 bias）进行操作。这些缩放和平移因子是通过训练过程中学习到的
        # 缩放平移之后才会进入激活函数
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    # 负责调用上面定义的那些函数，正式的前向传播过程
    def forward(self, x):
        batchsize = x.size()[0]# 表示 PyTorch 张量 x 的第一个维度的大小（即张量的行数或样本数），通常用于获取张量的批次大小（batch size）
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]# 将输入的x张量沿着第2维度n（1*1024*n）取最大值，并保留维度，返回的结果是一个张量，其大小与输入张量x相同，但每一行的值都被替换为该行的最大值；0是值，1是所在位置
        x = x.view(-1, 1024)# 通过最大池化将点云的每个点的n维特征池化成1维特征，形状变为(n_points, 1024, 1) -> (1, 1024, 1)，然后通过view操作将其转化为(1, 1024)的形状。这个view操作相当于去掉了一维。-1表示这一维度的大小可以根据其他维度的大小自动推导出来

        x = F.relu(self.bn4(self.fc1(x)))# 全链接、归一化、激活函数
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)# 256->9维

        # 为了让PointNet能够识别出这些旋转的点云，需要将每个点云沿着一个随机轴旋转一定角度，以扩增数据集。这样可以使网络在训练时具有旋转不变性
        iden = Variable(torch.from_numpy(np.eye(self.input_channels).flatten().astype(np.float32))).view(1, self.input_channels*self.input_channels).repeat(batchsize, 1)# 首先创建一个 1*9，将其转化为 PyTorch 的张量，并重复 batchsize 次，变成一个形状为 (batchsize, 9) 的张量
        if x.is_cuda:# 如果在 GPU 上运行，则将 iden 转移到 GPU 上
            iden = iden.cuda()
        x = x + iden # 将 iden 加到 x 上，相当于对 x 进行平移操作
        x = x.view(-1, self.input_channels, self.input_channels)# 将 x 从形状为 (batchsize, 1024) 转化为形状为 (batchsize, 3, 3) 的张量
        return x


class STNkd(nn.Module):#feature transform(T-Net)生成k=64*64矩阵  第二次姿态调整
    def __init__(self, k=64):
        super(STNkd, self).__init__()
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k*k)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.k = k

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))).view(1,self.k*self.k).repeat(batchsize,1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x

class PointNetfeat(nn.Module):#pointnet整个流程
    def __init__(self, input_channels = 3, global_feat = True, feature_transform = False):# 定义global_feat，用于了解点云的全局特征，例如点云的整体形状、姿态或全局分布等。表示是否需要进行特征变换，即学习一个变换矩阵用于调整点云的表示
        super(PointNetfeat, self).__init__()# 当子类继承自父类时，子类需要调用父类的初始化函数，以便继承父类的属性和方法，并进行必要的初始化操作。super()函数返回父类的对象，通过调用其 __init__() 方法，可以执行父类的初始化操作
        self.stn = STN3d(input_channels=input_channels) # 用一下TNet做姿态变换，矩阵大小不变
        self.conv1 = torch.nn.Conv1d(input_channels, 64, 1)# 一维卷积：输入3，输出64
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.global_feat = global_feat# 保存是否提取全局特征的标志，全局特征就是看整个点云数据的特征
        self.feature_transform = feature_transform# 保存是否进行特征变换的标志
        if self.feature_transform:
            self.fstn = STNkd(k=64)# 如果进行特征变换，创建一个k=64的TNet

    def forward(self, x):# 前向传播函数，接受输入x
        n_pts = x.size()[2]# 获取输入点云的点数，在第二维度上
        trans = self.stn(x)# 姿态变换，矩阵大小不变
        x = x.transpose(2, 1)# 将张量 x 的第二个维度和第三个维度进行互换，使得卷积层能够正确处理数据
        x = torch.bmm(x, trans)# 将变换后的点云与输入点云进行矩阵乘法
        x = x.transpose(2, 1)# 变回来，达到对点云数据进行空间变换的目的
        x = F.relu(self.bn1(self.conv1(x)))

        if self.feature_transform:# 如果进行特征变换
            trans_feat = self.fstn(x)# 对x用k=64的TNet
            x = x.transpose(2,1)# 再次交换维度
            x = torch.bmm(x, trans_feat)# 将变换后的特征与输入特征进行矩阵乘法
            x = x.transpose(2,1)
        else:
            trans_feat = None

        pointfeat = x # 将当前的特征保存在pointfeat(1*64*n)变量中，用于后续的拼接操作
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        if self.global_feat:
            return x, trans, trans_feat
        else:# 如果self.global_feat为False，则需要进行拼接操作
            x = x.view(-1, 1024, 1).repeat(1, 1, n_pts)# 第一个1表示在第1个维度上不进行重复复制，即保持原来的大小不变；第二个1表示在第2个维度上不进行重复复制，同样保持原来的大小不变；n_pts表示在第3个维度上进行n_pts次重复复制
            return torch.cat([x, pointfeat], 1), trans, trans_feat# 1*64*n和1*1024*n进行拼接得到1*64+1024*n，因为做了最大池化，所以n个点的信息会丢失局部信息而只有全局信息，所以做了一个拼接，保留了局部信息

class PointNetCls(nn.Module):#classification
    def __init__(self, k=2, feature_transform=False):
        super(PointNetCls, self).__init__()
        self.feature_transform = feature_transform
        self.feat = PointNetfeat(global_feat=True, feature_transform=feature_transform)# 创建整个pointnet框架
        # 定义三个全连接层，用于将提取的特征映射到最终的分类结果。每个全连接层的输入和输出维度依次为 1024 -> 512 -> 256 -> k
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k)
        self.dropout = nn.Dropout(p=0.3)# 输入的每个元素有 30% 的概率被置零
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()

    def forward(self, x):
        x, trans, trans_feat = self.feat(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1), trans, trans_feat# softmax就是相当于归一化然后给出不同类的概率


class PointNetDenseCls(nn.Module):#segmentation
    def __init__(self, k = 2, feature_transform=False):
        super(PointNetDenseCls, self).__init__()
        self.k = k
        self.feature_transform=feature_transform
        self.feat = PointNetfeat(global_feat=False, feature_transform=feature_transform)
        self.conv1 = torch.nn.Conv1d(1088, 512, 1)
        self.conv2 = torch.nn.Conv1d(512, 256, 1)
        self.conv3 = torch.nn.Conv1d(256, 128, 1)
        self.conv4 = torch.nn.Conv1d(128, self.k, 1)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)

    def forward(self, x):
        batchsize = x.size()[0]
        n_pts = x.size()[2]
        x, trans, trans_feat = self.feat(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.conv4(x)
        x = x.transpose(2,1).contiguous()
        x = F.log_softmax(x.view(-1,self.k), dim=-1)
        x = x.view(batchsize, n_pts, self.k)
        return x, trans, trans_feat

# 目的是计算特征变换矩阵（trans）的正交性损失。正交性损失用于约束特征变换矩阵的正交性，使得变换后的特征具有更好的表示能力
def feature_transform_regularizer(trans):
    d = trans.size()[1]# 特征维度的数量
    batchsize = trans.size()[0]# 批量大小
    I = torch.eye(d)[None, :, :]# 创建一个大小为 d x d 的单位矩阵，进行维度扩展，通过在第一个维度上添加 None，实现对单位矩阵的扩展，变成大小为 1 x d x d 的三维矩阵
    if trans.is_cuda:
        I = I.cuda()
    # 将 trans 和其转置矩阵相乘
    loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2,1)) - I, dim=(1,2)))
    return loss

if __name__ == '__main__':
    sim_data = Variable(torch.rand(2,3,525))# 模拟输入数据。这个张量表示一个具有 2 个样本，每个样本有 3 个特征，且每个特征有 525 个值
    trans = STN3d()
    out = trans(sim_data)
    print('stn', out.size())
    print('loss', feature_transform_regularizer(out))

    # sim_data_64d = Variable(torch.rand(32, 64, 2500))
    # trans = STNkd(k=64)
    # out = trans(sim_data_64d)
    # print('stn64d', out.size())
    # print('loss', feature_transform_regularizer(out))

    # pointfeat = PointNetfeat(global_feat=True)
    # out, _, _ = pointfeat(sim_data)
    # print('global feat', out.size())

    # pointfeat = PointNetfeat(global_feat=False)
    # out, _, _ = pointfeat(sim_data)
    # print('point feat', out.size())

    # cls = PointNetCls(k = 5)
    # out, _, _ = cls(sim_data)
    # print('class', out.size())

    # seg = PointNetDenseCls(k = 3)
    # out, _, _ = seg(sim_data)
    # print('seg', out.size())