"""
Test script for few-shot learning evaluation.

Evaluates trained models on few-shot classification tasks using
logistic regression on frozen features.
"""
import argparse
import os
import random
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm

from dataset import MiniTestDataset
from models.simsiam import SimSiam
from resnet_cmc import resnet50
from util import confidence_interval

parser = argparse.ArgumentParser(description='Few-Shot Learning Evaluation')

# Data parameters
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--dataset', default='miniimage',
                    help='dataset name: miniimage/tieredimage')
parser.add_argument('--data_folder', default='./mini_imagenet/',
                    help='path to dataset folder')
parser.add_argument('--model_path', default='./model/', type=str,
                    help='path to model directory')
parser.add_argument('--visual_dir', default='./log/visual/', type=str,
                    help='path to visualization directory')

# Model parameters
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                    help='model architecture (default: resnet50)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to checkpoint (default: none)')

# Data loading parameters
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                    help='number of data loading workers (default: 0)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256)')

# Optimization parameters (for fine-tuning if needed)
parser.add_argument('--lr', '--learning-rate', default=0.03, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum of SGD solver')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('--pe', '--pretrain_epoch', default=20, type=int,
                    help='pretrain on support set epochs')

# System parameters
parser.add_argument('--seed', default=111, type=int,
                    help='seed for initializing training')
parser.add_argument('--gpu', default=None, type=str,
                    help='GPU id to use')

# MoCo specific configs
parser.add_argument('--moco-dim', default=128, type=int,
                    help='feature dimension (default: 128)')
parser.add_argument('--moco-k', default=1280, type=int,
                    help='queue size; number of negative keys')
parser.add_argument('--moco-m', default=0.999, type=float,
                    help='moco momentum of updating key encoder')
parser.add_argument('--moco-t', default=0.07, type=float,
                    help='softmax temperature (default: 0.07)')

# Few-shot test configs
parser.add_argument('--task_num', default=1000, type=int,
                    help='number of testing tasks')
parser.add_argument('--n_way', default=5, type=int,
                    help='number of classes for test')
parser.add_argument('--train_way', default=64, type=int,
                    help='number of classes for training')
parser.add_argument('--k_shot', default=1, type=int,
                    help='number of support samples per class')
parser.add_argument('--k_query', default=15, type=int,
                    help='number of query samples per class')
parser.add_argument('--update_step', type=int, default=5,
                    help='task-level inner update steps')
parser.add_argument('--update_step_test', type=int, default=10,
                    help='update steps for fine-tuning')
parser.add_argument('--meta_lr', type=float, default=1e-3,
                    help='meta-level outer learning rate')
parser.add_argument('--update_lr', type=float, default=0.4,
                    help='task-level inner update learning rate')
parser.add_argument('--select_cls', default=None, nargs='+', type=int,
                    help='select specific classes for testing')

# MoCo v2 options
parser.add_argument('--mlp', action='store_true',
                    help='use mlp head')
parser.add_argument('--aug-plus', action='store_true',
                    help='use moco v2 data augmentation')
parser.add_argument('--cos', action='store_true',
                    help='use cosine lr schedule')


def main():
    """Main function to set up and run evaluation."""
    args = parser.parse_args()
    print(args)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    if args.seed is not None:
        print("Seed:", args.seed)
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        np.random.seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed evaluation. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down evaluation considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    main_worker(args)


def main_worker(args):
    """
    Worker function for model evaluation.

    Args:
        args: Command line arguments
    """
    # Create visualization directory if needed
    if not os.path.exists(args.visual_dir):
        os.makedirs(args.visual_dir)
        print("Created directory:", args.visual_dir)

    args.model = args.arch

    # Build model
    encoder = resnet50(num_classes=args.train_way, zero_init_residual=True)
    model = SimSiam(encoder, 2048, 512)
    model.cuda()
    normalize = Normalize(2)
    cudnn.benchmark = True

    # Load test dataset
    testdir = args.data
    test_dataset = MiniTestDataset(
        task_num=args.task_num,
        n_way=args.n_way,
        k_shot=args.k_shot,
        k_query=args.k_query
    )

    test_sampler = None

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        sampler=test_sampler,
        drop_last=True
    )

    model_path = args.resume

    # Load checkpoint
    if model_path:
        if os.path.isfile(model_path):
            print("=> Loading checkpoint '{}'".format(model_path))
            if args.gpu is None:
                checkpoint = torch.load(model_path)
            else:
                checkpoint = torch.load(
                    model_path, map_location={'cuda:0': 'cuda:0'}
                )
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['model'])
            print("=> Loaded checkpoint '{}' (epoch {})".format(
                model_path, checkpoint['epoch']))
        else:
            print("=> No checkpoint found at '{}'".format(model_path))
            raise FileNotFoundError("Checkpoint not found: {}".format(model_path))

    # Set model to evaluation mode
    model.eval()
    extractor = model.encoder
    extractor = torch.nn.Sequential(*(list(extractor.children())[:-3]))

    # Evaluate on test tasks
    corrects = []
    for data in tqdm(test_loader):
        x_spt, y_spt, x_qry, y_qry = data
        x_spt = x_spt.cuda()
        y_spt = y_spt
        x_qry = x_qry.cuda()
        y_qry = y_qry

        with torch.no_grad():
            # Extract features from support set
            feature = extractor(x_spt.squeeze()).squeeze().detach().cpu()

            # Train logistic regression classifier on support set
            clf = LogisticRegression(
                penalty='l2',
                random_state=0,
                C=1.0,
                solver='lbfgs',
                max_iter=1000,
                multi_class='multinomial'
            ).fit(normalize(feature), y_spt.squeeze())

            # Extract features from query set and predict
            q_feature = normalize(
                extractor(x_qry.squeeze()).squeeze().detach().cpu()
            )
            pred_q = clf.predict(q_feature)

            # Calculate accuracy
            correct = (torch.eq(
                torch.Tensor(pred_q), y_qry.squeeze()
            ).sum().item()) / pred_q.shape[0]

        corrects.append(correct)

    # Calculate statistics
    corrects = np.array(corrects)
    std = np.std(corrects, ddof=1)
    conf_interval = confidence_interval(std, args.task_num)

    print("Accuracy:", np.mean(corrects))
    print("Std:", std)
    print("Confidence interval:", conf_interval)


class Normalize(nn.Module):
    """Normalize features to unit length."""

    def __init__(self, power=2):
        """
        Initialize normalizer.

        Args:
            power: Power for norm calculation (default: 2 for L2 norm)
        """
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        """
        Normalize input tensor.

        Args:
            x: Input tensor

        Returns:
            Normalized tensor
        """
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out


if __name__ == '__main__':
    main()
