from __future__ import print_function

import torch
import os
import numpy as np
import math
from tqdm import tqdm
import matplotlib.pyplot as plt
from copy import deepcopy
import measures
from resnet_cmc import resnet50
from dataset import ImageRealFolderInstance
from dataset import  MiniUnlabelDataset
from models.simsiam import SimSiam
from torchvision import transforms
import torchvision.transforms.functional as F
import torchvision.models as models
import ipdb
import cv2

def adjust_learning_rate(epoch, opt, optimizer):
    """Sets the learning rate to the initial LR decayed by decay rate every steep step"""
    steps = np.sum(epoch > np.asarray(opt.lr_decay_epochs))
    if steps > 0:
        new_lr = opt.learning_rate * (opt.lr_decay_rate ** steps)
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res



def validate(model, device, val_loader, criterion):
    sum_loss, sum_correct = 0, 0
    margin = torch.Tensor([]).to(device)

    # switch to evaluation mode
    model.eval()
    with torch.no_grad():
        for i, (data, target) in enumerate(val_loader):
            data, target = data.to(device), target.to(device)

            # compute the output
            output = model(data)
            # output = torch.nn.functional.softmax(output, dim=1)

            # compute the classification error and loss
            pred = output.max(1)[1]
            sum_correct += pred.eq(target).sum().item()
            sum_loss += len(data) * criterion(output, target).item()

            # ipdb.set_trace()

            # compute the margin
            output_m = output.clone()
            for i in range(target.size(0)):
                output_m[i, target[i]] = output_m[i,:].min()
            margin = torch.cat((margin, output[:, target].diag() - output_m[:, output_m.max(1)[1]].diag()), 0)
        ipdb.set_trace()
        val_margin = np.percentile( margin.cpu().numpy(), 10 )

    return 1 - (sum_correct / len(val_loader.dataset)), sum_loss / len(val_loader.dataset), val_margin

def mixup_data(x, y, alpha, num_classes):

    '''Compute the mixup data. Return mixed inputs, pairs of targets, and lambda'''
    if alpha > 0.:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).cuda()
    mixed_x = lam * x + (1 - lam) * x[index,:]
    y_a, y_b = torch.nn.functional.one_hot(y, num_classes=num_classes), torch.nn.functional.one_hot(y[index], num_classes=num_classes)
    y_mix = lam * y_a + (1 - lam) * y_b
    return mixed_x, y_mix, lam


def returnCAM_tensor(feature_conv, weight_softmax, class_idx, size_upsample=(84,84), use_gpu=False):
    # generate the class activation maps upsample to size_upsample
    transform = transforms.Compose([
                    # transforms.ToPILImage(),
                    # transforms.Resize(size_upsample),
                    transforms.ToTensor(),
                    ])
    # bz, nc, h, w = feature_conv.shape
    nc, h, w = feature_conv.shape
    i=0
    for idx in class_idx:
        cam = torch.mm(weight_softmax[idx].reshape(1,weight_softmax[idx].shape[0]), feature_conv.reshape((nc, h*w))).data
        cam = cam.reshape(h, w)
        cam = cam - torch.min(cam)
        cam_img = cam / torch.max(cam)
        cam_img = (255 * cam_img.cpu().numpy())
        output_cam = (transform(cv2.resize(cam_img, size_upsample))/255.0).unsqueeze(0)
    return output_cam.cuda()



def confidence_interval(std, n):
    return 1.92 * std / np.sqrt(n)

    
if __name__ == '__main__':
    meter = AverageMeter()
                                                                              