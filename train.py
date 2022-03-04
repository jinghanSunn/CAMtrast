"""
Training MoCo and Instance Discrimination

InsDis: Unsupervised feature learning via non-parametric instance discrimination
MoCo: Momentum Contrast for Unsupervised Visual Representation Learning
Feature Projection

"""
from __future__ import print_function
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import os
import sys
import time
import torch
import torch.backends.cudnn as cudnn
import argparse
import socket
import math
import numpy as np
import cv2

# import tensorboard_logger as tb_logger

from torchvision import transforms, datasets
import torchvision.models as models
import torch.optim as optim
from torchtoolbox.transform import Cutout
from util import  AverageMeter, mixup_data, adjust_learning_rate
from PIL import Image

from models.resnet import InsResNet50Sp
from NCE.NCEAverage import MemoryInsDis
from NCE.NCEAverage import MemoryMoCo
from NCE.NCECriterion import NCECriterion
from NCE.NCECriterion import NCESoftmaxLoss

from dataset import ImageRealFolderInstance
from resnet_cmc import resnet50
from models.simsiam import SimSiam

import moco.builder
import moco.loader

from util import returnCAM_tensor

try:
    from apex import amp, optimizers
except ImportError:
    pass
"""
TODO: python 3.6 ModuleNotFoundError
"""
import warnings
warnings.filterwarnings("ignore")

def parse_option():

    hostname = socket.gethostname()

    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=10, help='print frequency')
    parser.add_argument('--tb_freq', type=int, default=500, help='tb frequency')
    parser.add_argument('--save_freq', type=int, default=10, help='save frequency')
    parser.add_argument('--batch_size', type=int, default=64, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=18, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=240, help='number of training epochs')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.05, help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='60,80', help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')

    # crop
    parser.add_argument('--crop', type=float, default=0.2, help='minimum crop')

    # dataset
    parser.add_argument('--dataset', type=str, default='imagenet100', choices=['imagenet100', 'imagenet', 'tieredimage', 'cifar', 'imagenet'])
    parser.add_argument('--data_folder', type=str, default='./mini_imagenet')
    parser.add_argument('--n_way', type=int, default=64)
    parser.add_argument('--image_num', type=int, default=1300, help='each class train images')
    # resume
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')

    # augmentation setting
    parser.add_argument('--aug', type=str, default='CJ', choices=['NULL', "cjv2"])

    # warm up
    parser.add_argument('--warm', action='store_true', help='add warm-up setting')
    parser.add_argument('--amp', action='store_true', help='using mixed precision')
    parser.add_argument('--opt_level', type=str, default='O2', choices=['O1', 'O2'])

    # model definition
    parser.add_argument('--model', type=str, default='resnet50', choices=['resnet50', 'resnet50st','resnet50x2', 'resnet50x4'])
    parser.add_argument('--model_name', type=str)

    # loss function
    parser.add_argument('--softmax', action='store_true', help='using softmax contrastive loss rather than NCE')
    parser.add_argument('--nce_k', type=int, default=16384)
    parser.add_argument('--nce_t', type=float, default=0.07)
    parser.add_argument('--nce_m', type=float, default=0.5)

    # memory setting
    parser.add_argument('--moco', action='store_true', help='using MoCo (otherwise Instance Discrimination)')
    parser.add_argument('--alpha', type=float, default=0.999, help='exponential moving average weight')

    # GPU setting
    parser.add_argument('--gpu', default=None, type=int, help='GPU id to use.')

    # loss setting
    parser.add_argument('--unif', type=float, default=0, help='weight for uniform loss')
    parser.add_argument('--mixup', action='store_true', help='manifold mixup')
    parser.add_argument('--mix_alpha', type=float, default=1.0, help='manifold mixup alpha')
    parser.add_argument('--layer_mix', type=int, default=None, help='which layer to mix')
    
    # heatmap setting
    parser.add_argument('--epoch_t', type=int, default=100, help='epoch threshold')
    parser.add_argument('--cam_mode', type=str, default='reverse', help='heatmap process mode')
    parser.add_argument('--cam_t', type=float, default=0.5, help="heatmap process threshold, if mode=='hard_thresh'")
    parser.add_argument('--cam_momentum', action='store_true', help='momentum update heatmap')
    parser.add_argument('--cam_k', type=float, default=0.9, help="momentum update heatmap ratio")
    parser.add_argument('--cam_aug', action='store_true', help='augment after cam aug')

    # simsiam specific configs:
    parser.add_argument('--dim', default=2048, type=int,
                    help='feature dimension (default: 2048)')
    parser.add_argument('--pred-dim', default=512, type=int,
                    help='hidden dimension of the predictor (default: 512)')

    opt = parser.parse_args()

    # set the path according to the environment
    # if hostname.startswith('visiongpu'):
    opt.model_path = '/model'
    opt.tb_path = '/model'

    if opt.dataset == 'imagenet':
        if 'alexnet' not in opt.model:
            opt.crop = 0.08

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.method = 'softmax' if opt.softmax else 'nce'

    print(opt)
    print(opt.model_name)

    opt.model_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.model_folder):
        os.makedirs(opt.model_folder)

    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)
    print("save model to", opt.model_folder)
    return opt


def moment_update(model, model_ema, m):
    """ model_ema = m * model_ema + (1 - m) model """
    for p1, p2 in zip(model.parameters(), model_ema.parameters()):
        p2.data.mul_(m).add_(1-m, p1.detach().data)


def get_shuffle_ids(bsz):
    """generate shuffle ids for ShuffleBN"""
    forward_inds = torch.randperm(bsz).long().cuda()
    backward_inds = torch.zeros(bsz).long().cuda()
    value = torch.arange(bsz).long().cuda()
    backward_inds.index_copy_(0, forward_inds, value)
    return forward_inds, backward_inds


def main():

    args = parse_option()

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    # set the data loader
    if args.dataset == 'imagenet100':
        data_folder = os.path.join(args.data_folder, 'train')
    else:
        data_folder = args.data_folder

    image_size = 84
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    normalize = transforms.Normalize(mean=mean, std=std)

    train_transform = transforms.Compose([
    transforms.RandomResizedCrop(image_size, scale=(args.crop, 1.)),
    transforms.RandomGrayscale(p=0.2),
    transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
    transforms.RandomApply([moco.loader.GaussianBlur([.1, 2.])], p=0.5),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize,
        ])
    

    train_dataset = ImageRealFolderInstance('train', transform=train_transform, return_idx=True)

    print(len(train_dataset))
    train_sampler = None
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, sampler=train_sampler)

    # create model and optimizer
    n_data = len(train_dataset)

    if args.model == 'resnet50':
        encoder = resnet50(num_classes=args.n_way, zero_init_residual=True)
        if args.moco:
            model_ema = resnet50()
    else:
        raise NotImplementedError('model not supported {}'.format(args.model))
    
    model = SimSiam(encoder, args.dim, args.pred_dim)
    encoder = model.encoder
    model = torch.nn.DataParallel(model)
    encoder = torch.nn.DataParallel(encoder)


    # copy weights from `model' to `model_ema'
    if args.moco:
        moment_update(model, model_ema, 0)
    

    # set the contrast memory and criterion
    if args.moco:
        contrast = MemoryMoCo(128, n_data, args.nce_k, args.nce_t, args.softmax).cuda(args.gpu)
    else:
        contrast = MemoryInsDis(128, n_data, args.nce_k, args.nce_t, args.nce_m, args.softmax).cuda(args.gpu)

    criterion = torch.nn.CrossEntropyLoss()
    criterion = criterion.cuda(args.gpu)
    criterion_simsiam = torch.nn.CosineSimilarity(dim=1).cuda(args.gpu)

    model = model.cuda()

    optim_params = [{'params': model.module.encoder.parameters(), 'fix_lr': False},
                        {'params': model.module.predictor.parameters(), 'fix_lr': True}]

    optimizer = torch.optim.SGD(optim_params,
                                lr=args.learning_rate,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    cudnn.benchmark = True

    if args.amp:
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.opt_level)
        if args.moco:
            optimizer_ema = torch.optim.SGD(model_ema.parameters(),
                                            lr=0,
                                            momentum=0,
                                            weight_decay=0)
            model_ema, optimizer_ema = amp.initialize(model_ema, optimizer_ema, opt_level=args.opt_level)

    # optionally resume from a checkpoint
    args.start_epoch = 1
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location='cpu')
            args.start_epoch = checkpoint['epoch'] + 1
            model.load_state_dict(checkpoint['model'])
            if args.moco:
                model_ema.load_state_dict(checkpoint['model_ema'])

            if args.amp and checkpoint['opt'].amp:
                print('==> resuming amp state_dict')
                amp.load_state_dict(checkpoint['amp'])

            print("=> loaded successfully '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
            del checkpoint
            torch.cuda.empty_cache()
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    for epoch in range(args.start_epoch, args.epochs + 1):

        adjust_learning_rate(epoch, args, optimizer)
        print("==> training...")

        time1 = time.time()
        loss, prob = train(epoch, train_loader, model, contrast, criterion, criterion_simsiam, optimizer, args)
        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

        # save model
        if epoch % args.save_freq == 0:
            print('==> Saving...')
            state = {
                'opt': args,
                'model': model.state_dict(),
                'contrast': contrast.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
            }
            if args.moco:
                state['model_ema'] = model_ema.state_dict()
            if args.amp:
                state['amp'] = amp.state_dict()
            save_file = os.path.join(args.model_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            torch.save(state, save_file)
            print("save model to", save_file)
            # help release GPU memory
            del state
            torch.cuda.empty_cache()


def train(epoch, train_loader, model, contrast, criterion, criterion_simsiam, optimizer, opt):
    """
    one epoch training for instance discrimination
    """
    model.train()

    def set_bn_train(m):
        classname = m.__class__.__name__
        if classname.find('BatchNorm') != -1:
            m.train()
    # model_ema.apply(set_bn_train)

    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    prob_meter = AverageMeter()

    end = time.time()

    for idx, (ori_img, inputs, y) in enumerate(train_loader):
        data_time.update(time.time() - end)

        bsz = inputs.size(0)

        # ori_img = ori_img.float()
        inputs = inputs.float()
        if opt.gpu is not None:
            inputs = inputs.cuda(opt.gpu, non_blocking=True)
        else:
            inputs = inputs.cuda()
        y = y.cuda()
        # ===================forward=====================

        # ids for ShuffleBN
        shuffle_ids, reverse_ids = get_shuffle_ids(bsz)
        
        _, output1, feature = model.encoder(inputs, notsimsiam=True)
        
        loss = criterion(output1, y.long())
        prob = (torch.eq(output1.argmax(dim=1), y).sum().item())/y.shape[0]


        # ===================backward=====================
        optimizer.zero_grad()
        if opt.amp:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        optimizer.step()

        # ===================meters=====================
        loss_meter.update(loss.item(), bsz)
        prob_meter.update(prob, bsz)

        # moment_update(model, model_ema, opt.alpha)

        torch.cuda.synchronize()
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if (idx + 1) % opt.print_freq == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})\t'
                  'prob {prob.val:.3f} ({prob.avg:.3f})'.format(
                   epoch, idx + 1, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=loss_meter, prob=prob_meter))
            # print(out.shape)
            sys.stdout.flush()
        
        # ===================new anti-cam train===================== # 
        if epoch>=opt.epoch_t:
            antiCAMs = getCAM(inputs, output1, feature, model.encoder, opt)

            # ===================new anti-cam forward=====================
            p1, p2, z1, z2, _, anti_output = model(x1=inputs, x2=antiCAMs)
            sim_loss = -(criterion_simsiam(p1, z2).mean() + criterion_simsiam(p2, z1).mean()) * 0.5
            anti_loss = criterion(anti_output, y.long())
            loss = anti_loss + sim_loss
            prob = (torch.eq(anti_output.argmax(dim=1), y).sum().item())/y.shape[0]

            # ===================backward=====================
            optimizer.zero_grad()
            if opt.amp:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            optimizer.step()

            # ===================meters=====================
            loss_meter.update(loss.item(), bsz)
            prob_meter.update(prob, bsz)

            # moment_update(model, model_ema, opt.alpha)

            torch.cuda.synchronize()
            batch_time.update(time.time() - end)
            end = time.time()

            # print info
            if (idx + 1) % opt.print_freq == 0:
                print('Train: [{0}][{1}/{2}]\t'
                    'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    'loss {loss.val:.3f} ({loss.avg:.3f})\t'
                    'prob {prob.val:.3f} ({prob.avg:.3f})'.format(
                    epoch, idx + 1, len(train_loader), batch_time=batch_time,
                    data_time=data_time, loss=loss_meter, prob=prob_meter))
                # print(out.shape)
                sys.stdout.flush()
        

    return loss_meter.avg, prob_meter.avg


def getCAM(inputs, output1, feature, model, opt, train_transform=None):
    logits = output1.detach() # [bs, n_way]
    features_blobs = feature.detach()
    params = list(model.parameters())[-2] 
    weight_softmax = params.detach().data.squeeze() # [n_way, nce_k]
    if opt.dataset == 'miniimagenet' or opt.dataset == 'tieredimage' or opt.dataset == 'imagenet100':
        size_upsample = (84,84)

    for i, logit in enumerate(logits):
        h_x = torch.nn.functional.softmax(logit, dim=0).data
        probs, idx_cam = h_x.sort(0, True)


        CAM = returnCAM_tensor(features_blobs[i], weight_softmax, [idx_cam[0]], size_upsample=size_upsample, use_gpu=True)
        if i == 0:
            antiCAMs = torch.mul(antiCamFunction(CAM, mode=opt.cam_mode, t=opt.cam_t), inputs[i])
        else:
            antiCAMs = torch.cat([antiCAMs, torch.mul(antiCamFunction(CAM,mode=opt.cam_mode,t=opt.cam_t), inputs[i])], dim=0)
    return antiCAMs


def antiCamFunction(CAM, mode='reverse', t=None):
    if mode == 'reverse':
        return 1-CAM[0].unsqueeze(0)

    
def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2



def uniform_loss(x, t=2):
  return torch.pdist(x, p=2).pow(2).mul(-t).exp().mean().log()


if __name__ == '__main__':
    main()
