import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
eps = 1e-7


class NCECriterion(nn.Module):
    """
    Eq. (12): L_{NCE}
    """
    def __init__(self, n_data):
        super(NCECriterion, self).__init__()
        self.n_data = n_data

    def forward(self, x):
        bsz = x.shape[0]
        m = x.size(1) - 1

        # noise distribution
        Pn = 1 / float(self.n_data)

        # loss for positive pair
        P_pos = x.select(1, 0)
        log_D1 = torch.div(P_pos, P_pos.add(m * Pn + eps)).log_()

        # loss for K negative pair
        P_neg = x.narrow(1, 1, m)
        log_D0 = torch.div(P_neg.clone().fill_(m * Pn), P_neg.add(m * Pn + eps)).log_()

        loss = - (log_D1.sum(0) + log_D0.view(-1, 1).sum(0)) / bsz

        return loss


class NCESoftmaxLoss(nn.Module):
    """Softmax cross-entropy loss (a.k.a., info-NCE loss in CPC paper)"""
    def __init__(self):
        super(NCESoftmaxLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        bsz = x.shape[0]
        x = x.squeeze()
        label = torch.zeros([bsz]).cuda().long()
        loss = self.criterion(x, label)
        return loss



class DistillKL(nn.Module):
    """KL divergence for distillation"""
    def __init__(self, T):
        super(DistillKL, self).__init__()
        self.T = T

    def forward(self, y_s, y_t):
        p_s = F.log_softmax(y_s/self.T, dim=1)
        p_t = F.softmax(y_t/self.T, dim=1)
        loss = F.kl_div(p_s, p_t, reduction='sum') * (self.T**2) / y_s.shape[0]
        return loss


def CrossEntropyLoss_label_smooth( targets, num_classes=10, epsilon=0.1):
    N = len(targets)
    print(N)
    print(num_classes)
    # targets = torch.nn.functional.one_hot(torch.Tensor(targets).long(), num_classes=num_classes)
    targets = torch.Tensor(targets).long().unsqueeze(1)
    print(targets.shape)
    # torch.Size([8, 10])
    # 初始化一个矩阵, 里面的值都是epsilon / (num_classes - 1)
    smoothed_labels = torch.full(size=(N, num_classes), fill_value=epsilon / (num_classes - 1))
    print(smoothed_labels.shape)
    # 为矩阵中的每一行的某个index的位置赋值为1 - epsilon
    smoothed_labels.scatter_(dim=1, index=targets, value=1 - epsilon)
    # 调用torch的log_softmax
    # log_prob = nn.functional.log_softmax(outputs, dim=1)
    # 用之前得到的smoothed_labels来调整log_prob中每个值
    # loss = - torch.sum(log_prob * smoothed_labels) / N
    print(smoothed_labels[:10])
    print(smoothed_labels[-10:])
    return smoothed_labels
