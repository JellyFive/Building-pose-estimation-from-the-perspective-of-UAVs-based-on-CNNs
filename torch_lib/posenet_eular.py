import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from tensorboardX import SummaryWriter


def OrientationLoss(orient_batch, orientGT_batch, confGT_batch):

    batch_size = orient_batch.size()[0]
    indexes = torch.max(confGT_batch, dim=1)[1]

    # extract just the important bin
    orientGT_batch = orientGT_batch[torch.arange(batch_size), indexes]
    orient_batch = orient_batch[torch.arange(batch_size), indexes]

    theta_diff = torch.atan2(orientGT_batch[:, 1], orientGT_batch[:, 0])
    estimated_theta_diff = torch.atan2(orient_batch[:, 1], orient_batch[:, 0])

    return -1 * torch.cos(theta_diff - estimated_theta_diff).mean()


class Model(nn.Module):
    def __init__(self, features=None, bins=8, w=1):  # 神经网络的基本结构
        super(Model, self).__init__()  # 先运行nn.Module的初始化函数
        self.bins = bins
        self.w = w
        self.features = features
        # self.features = nn.Sequential(*list(features.children())[:-1])
        self.orientation = nn.Sequential(
            # nn.Linear(512, 512),  # resnet101
            nn.Linear(1000, 512),  # mobilenetv2
            # nn.Linear(512 * 7 * 7, 256),  # 全连接函数，将512*7*7连接到256个节点上 vgg19
            nn.ReLU(True),  # 激活函数
            nn.Dropout(),
            nn.Linear(512, 256),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(256, bins*2)  # to get sin and cos # 角度回归
        )
        self.confidence = nn.Sequential(
            nn.Linear(1000, 512),  # mobilenetv2
            # nn.Linear(512, 512),  # resnet101
            # nn.Linear(512 * 7 * 7, 256),  # vgg19
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 256),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(256, bins),
            # nn.Softmax()
            # nn.Sigmoid()
        )

        self.orientation_yaw = nn.Sequential(
            # nn.Linear(512, 512),  # resnet101
            nn.Linear(1000, 512),  # mobilenetv2
            # nn.Linear(512 * 7 * 7, 256),  # 全连接函数，将512*7*7连接到256个节点上 vgg19
            nn.ReLU(True),  # 激活函数
            nn.Dropout(),
            nn.Linear(512, 256),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(256, bins*2)  # to get sin and cos # 角度回归
        )
        self.confidence_yaw = nn.Sequential(
            nn.Linear(1000, 512),  # mobilenetv2
            # nn.Linear(512, 512),  # resnet101
            # nn.Linear(512 * 7 * 7, 256),  # vgg19
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 256),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(256, bins),
            # nn.Softmax()
            # nn.Sigmoid()
        )

    def forward(self, x):  # 前向传播函数，向后传播函数自动生成autograd
        x = self.features(x)  # 512 x 7 x 7
        # view函数将张量x形成一维的向量形式，总特征数是512*7*7，用于后面的全连接层 vgg19
        # x = x.view(-1, 512 * 7 * 7)
        x = x.view(x.size(0), -1)  # resnet101
        orientation = self.orientation(x)
        orientation = orientation.view(-1, self.bins, 2)  # 先对角度进行两次全连接再进行归一化
        # 对输入的数据（tensor）进行指定维度的L2_norm运算。
        orientation = F.normalize(orientation, dim=2)
        confidence = self.confidence(x)

        orientation_yaw = self.orientation_yaw(x)
        orientation_yaw = orientation_yaw.view(-1, self.bins, 2)
        orientation_yaw = F.normalize(orientation_yaw, dim=2)
        confidence_yaw = self.confidence_yaw(x)
        return orientation, confidence, orientation_yaw, confidence_yaw
