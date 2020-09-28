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


class FocalLoss(nn.Module):
    r"""
        This criterion is a implemenation of Focal Loss, which is proposed in 
        Focal Loss for Dense Object Detection.

            Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])

        The losses are averaged across observations for each minibatch.

        Args:
            alpha(1D Tensor, Variable) : the scalar factor for this criterion
            gamma(float, double) : gamma > 0; reduces the relative loss for well-classiﬁed examples (p > .5), 
                                   putting more focus on hard, misclassiﬁed examples
            size_average(bool): By default, the losses are averaged over observations for each minibatch.
                                However, if the field size_average is set to False, the losses are
                                instead summed for each minibatch.


    """

    def __init__(self, bins=4, alpha=None, gamma=2, size_average=True):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(bins, 1))
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(alpha)
        self.gamma = gamma
        self.bins = bins
        self.size_average = size_average

    def forward(self, inputs, targets):
        N = inputs.size(0)
        C = inputs.size(1)
        P = F.softmax(inputs)

        class_mask = inputs.data.new(N, C).fill_(0)
        class_mask = Variable(class_mask)
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.)
        # print(class_mask)

        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()
        alpha = self.alpha[ids.data.view(-1)]

        probs = (P*class_mask).sum(1).view(-1, 1)

        log_p = probs.log()
        #print('probs size= {}'.format(probs.size()))
        # print(probs)

        batch_loss = -alpha*(torch.pow((1-probs), self.gamma))*log_p
        # print('-----bacth_loss------')
        # print(batch_loss)

        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss


class Model(nn.Module):
    def __init__(self, base_model=None, bins=8):  # 神经网络的基本结构
        super(Model, self).__init__()  # 先运行nn.Module的初始化函数

        # feat_in = base_model.fc.in_features
        # self.base_model = nn.Sequential(*list(base_model.children())[:-1])
        self.bins = bins
        self.base_model = base_model

        # self.fc_rotation = nn.Linear(1000, 4, bias=False)

        self.location = nn.Sequential(
            # nn.Linear(512, 512),  # resnet101
            nn.Linear(1000, 512),  # mobilenetv2
            # nn.Linear(512 * 7 * 7, 256),  # 全连接函数，将512*7*7连接到256个节点上 vgg19
            nn.ReLU(True),  # 激活函数
            nn.Dropout(),
            nn.Linear(512, 256),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(256, 3)  # to get sin and cos # 角度回归
        )

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


    def forward(self, x):
        x = self.base_model(x)
        x = x.view(x.size(0), -1)
        location = self.location(x)
        orientation = self.orientation(x)
        orientation = orientation.view(-1, self.bins, 2)  # 先对角度进行两次全连接再进行归一化
        # 对输入的数据（tensor）进行指定维度的L2_norm运算。
        orientation = F.normalize(orientation, dim=2)
        confidence = self.confidence(x)

        orientation_yaw = self.orientation_yaw(x)
        orientation_yaw = orientation_yaw.view(-1, self.bins, 2)
        orientation_yaw = F.normalize(orientation_yaw, dim=2)
        confidence_yaw = self.confidence_yaw(x)

        return location, orientation, confidence, orientation_yaw, confidence_yaw
        # return location
