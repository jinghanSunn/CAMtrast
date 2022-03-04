import argparse
import builtins
import math
import os
import random
import shutil
import time
import warnings
from copy import deepcopy
from itertools import chain
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression


# import ipdb
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from models.simsiam import SimSiam


from models.resnet import InsResNet50Sp
from dataset import MiniTestDataset
from resnet_cmc import resnet50
from util import  confidence_interval

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--dataset', default='miniimage',
                    help='miniimage/tieredimage')
parser.add_argument('--data_folder', default='./mini_imagenet/',
                    help='path to dataset')
parser.add_argument('--model_path', default='./model/', type=str,
                    help='path to dataset')
parser.add_argument('--visual_dir', default='./log/visual/', type=str,
                    help='path to visual')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                    # choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet50)')
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.03, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum of SGD solver')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('--pe', '--pretrain_epoch', default=20, type=int,
                    help='pretrain on support set epoch')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--seed', default=111, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=str,
                    help='GPU id to use.')

# moco specific configs:
parser.add_argument('--moco-dim', default=128, type=int,
                    help='feature dimension (default: 128)')
parser.add_argument('--moco-k', default=1280, type=int,
                    help='queue size; number of negative keys (default: 65536)')
parser.add_argument('--moco-m', default=0.999, type=float,
                    help='moco momentum of updating key encoder (default: 0.999)')
parser.add_argument('--moco-t', default=0.07, type=float,
                    help='softmax temperature (default: 0.07)')

# test sest configs:
parser.add_argument('--task_num', default=1000, type=int,
                    help='number for testing tasks. ')
parser.add_argument('--n_way', default=5, type=int,
                    help='classes for test. ')
parser.add_argument('--train_way', default=64, type=int,
                    help='classes for test. ')
parser.add_argument('--k_shot', default=1, type=int)
parser.add_argument('--k_query', default=15, type=int)
parser.add_argument('--update_step', type=int, help='task-level inner update steps', default=5)
parser.add_argument('--update_step_test', type=int, help='update steps for finetunning', default=10)
parser.add_argument('--meta_lr', type=float, help='meta-level outer learning rate', default=1e-3)
parser.add_argument('--update_lr', type=float, help='task-level inner update learning rate', default=0.4)
parser.add_argument('--select_cls', default=None, nargs='+', type=int)

# options for moco v2
parser.add_argument('--mlp', action='store_true', 
                    help='use mlp head')
parser.add_argument('--aug-plus', action='store_true',
                    help='use moco v2 data augmentation')
parser.add_argument('--cos', action='store_true',
                    help='use cosine lr schedule')


def main():
    args = parser.parse_args()
    print(args)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    if args.seed is not None:
        print("seed", args.seed)
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        np.random.seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')
    
    main_worker(args)


def main_worker(args):

    if not os.path.exists(args.visual_dir):
        os.mkdir(args.visual_dir)
        print("mkdir", args.visual_dir)

    args.model = args.arch

    encoder = resnet50(num_classes=args.train_way, zero_init_residual=True)
    model = SimSiam(encoder, 2048, 512)
    model.cuda()
    normalize = Normalize(2)
    cudnn.benchmark = True

    # Data loading code
    testdir = args.data
    test_dataset = MiniTestDataset(task_num=args.task_num, n_way=args.n_way, k_shot=args.k_shot, k_query=args.k_query)

    test_sampler = None

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=int(1), shuffle=False,
        num_workers=args.workers, pin_memory=True, sampler=test_sampler, drop_last=True)


    model_path = args.resume
    
    # optionally resume from a checkpoint
    if model_path:
        if os.path.isfile(model_path):
            print("=> loading checkpoint '{}'".format(model_path))
            if args.gpu is None:
                checkpoint = torch.load(model_path)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(model_path, map_location={'cuda:0': 'cuda:0'}) # , map_location={'cuda:2':loc}
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['model'])
            print("=> loaded checkpoint '{}' (epoch {})"
                .format(model_path, checkpoint['epoch']))
            success = True
        else:
            print("=> no checkpoint found at '{}'".format(model_path))
            raise NotImplementedError

    
    model.eval()
    extractor = model.encoder
    extractor = torch.nn.Sequential(*(list(extractor.children())[:-3]))


    corrects = []
    for data in tqdm(test_loader):
        x_spt, y_spt, x_qry, y_qry = data
        x_spt = x_spt.cuda()
        y_spt = y_spt
        x_qry = x_qry.cuda()
        y_qry = y_qry
        with torch.no_grad():
            feature = extractor(x_spt.squeeze()).squeeze().detach().cpu()
            feature = feature.detach().cpu()
            clf = LogisticRegression(penalty='l2',
                                random_state=0,
                                C=1.0,
                                solver='lbfgs',
                                max_iter=1000,
                                multi_class='multinomial').fit(normalize(feature), y_spt.squeeze())

            q_feature = normalize(extractor(x_qry.squeeze()).squeeze().detach().cpu())
            pred_q = clf.predict(q_feature)
            
            prob_q = clf.predict_proba(q_feature)
            correct = (torch.eq(torch.Tensor(pred_q), y_qry.squeeze()).sum().item())/pred_q.shape[0]


    
        corrects.append(correct) 


    corrects = np.array(corrects)
    std = np.std(corrects, ddof = 1)
    conf_intveral = confidence_interval(std, args.task_num)
    
    print("accuracy:", np.mean(corrects))
    print("std:", std)
    print("confidence interval:",conf_intveral)
            

        
        

class Normalize(nn.Module):

    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out


if __name__ == '__main__':
    main()
