"""
Training MoCo and Instance Discrimination

InsDis: Unsupervised feature learning via non-parametric instance discrimination
MoCo: Momentum Contrast for Unsupervised Visual Representation Learning
Feature Projection
"""
from __future__ import print_function

import argparse
import os
import sys
import time
import warnings

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from PIL import ImageFile
from torchvision import transforms

from dataset import ImageRealFolderInstance
from models.simsiam import SimSiam
from NCE.NCEAverage import MemoryInsDis, MemoryMoCo
from resnet_cmc import resnet50
from util import AverageMeter, adjust_learning_rate, returnCAM_tensor
import moco.loader

# Allow loading of truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Suppress warnings
warnings.filterwarnings("ignore")

try:
    from apex import amp
except ImportError:
    amp = None

def parse_option():
    """Parse command line arguments for training configuration."""
    parser = argparse.ArgumentParser('argument for training')

    # Training parameters
    parser.add_argument('--print_freq', type=int, default=10,
                        help='print frequency')
    parser.add_argument('--tb_freq', type=int, default=500,
                        help='tensorboard frequency')
    parser.add_argument('--save_freq', type=int, default=10,
                        help='save frequency')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='batch size')
    parser.add_argument('--num_workers', type=int, default=18,
                        help='number of workers to use')
    parser.add_argument('--epochs', type=int, default=240,
                        help='number of training epochs')

    # Optimization parameters
    parser.add_argument('--learning_rate', type=float, default=0.05,
                        help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='60,80',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1,
                        help='decay rate for learning rate')
    parser.add_argument('--beta1', type=float, default=0.5,
                        help='beta1 for adam')
    parser.add_argument('--beta2', type=float, default=0.999,
                        help='beta2 for Adam')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')

    # Data augmentation
    parser.add_argument('--crop', type=float, default=0.2,
                        help='minimum crop')
    parser.add_argument('--aug', type=str, default='CJ',
                        choices=['NULL', "cjv2"],
                        help='augmentation setting')

    # Dataset parameters
    parser.add_argument('--dataset', type=str, default='imagenet100',
                        choices=['imagenet100', 'imagenet', 'tieredimage', 'cifar'],
                        help='dataset name')
    parser.add_argument('--data_folder', type=str, default='./mini_imagenet',
                        help='path to dataset folder')
    parser.add_argument('--n_way', type=int, default=64,
                        help='number of classes')
    parser.add_argument('--image_num', type=int, default=1300,
                        help='number of images per class for training')

    # Resume training
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')

    # Training optimization
    parser.add_argument('--warm', action='store_true',
                        help='add warm-up setting')
    parser.add_argument('--amp', action='store_true',
                        help='use mixed precision training')
    parser.add_argument('--opt_level', type=str, default='O2',
                        choices=['O1', 'O2'],
                        help='apex optimization level')

    # Model definition
    parser.add_argument('--model', type=str, default='resnet50',
                        choices=['resnet50', 'resnet50st', 'resnet50x2', 'resnet50x4'],
                        help='model architecture')
    parser.add_argument('--model_name', type=str,
                        help='name for saving model')

    # Loss function parameters
    parser.add_argument('--softmax', action='store_true',
                        help='use softmax contrastive loss rather than NCE')
    parser.add_argument('--nce_k', type=int, default=16384,
                        help='number of negative samples for NCE')
    parser.add_argument('--nce_t', type=float, default=0.07,
                        help='temperature for NCE')
    parser.add_argument('--nce_m', type=float, default=0.5,
                        help='momentum for NCE')

    # Memory bank settings
    parser.add_argument('--moco', action='store_true',
                        help='use MoCo (otherwise Instance Discrimination)')
    parser.add_argument('--alpha', type=float, default=0.999,
                        help='exponential moving average weight')

    # GPU setting
    parser.add_argument('--gpu', default=None, type=int,
                        help='GPU id to use')

    # Additional loss settings
    parser.add_argument('--unif', type=float, default=0,
                        help='weight for uniform loss')
    parser.add_argument('--mixup', action='store_true',
                        help='use manifold mixup')
    parser.add_argument('--mix_alpha', type=float, default=1.0,
                        help='manifold mixup alpha')
    parser.add_argument('--layer_mix', type=int, default=None,
                        help='which layer to mix')

    # CAM (Class Activation Map) settings
    parser.add_argument('--epoch_t', type=int, default=100,
                        help='epoch threshold for CAM')
    parser.add_argument('--cam_mode', type=str, default='reverse',
                        help='heatmap process mode')
    parser.add_argument('--cam_t', type=float, default=0.5,
                        help="heatmap process threshold, if mode=='hard_thresh'")
    parser.add_argument('--cam_momentum', action='store_true',
                        help='use momentum update for heatmap')
    parser.add_argument('--cam_k', type=float, default=0.9,
                        help='momentum update heatmap ratio')
    parser.add_argument('--cam_aug', action='store_true',
                        help='augment after CAM augmentation')

    # SimSiam specific configs
    parser.add_argument('--dim', default=2048, type=int,
                        help='feature dimension (default: 2048)')
    parser.add_argument('--pred-dim', default=512, type=int,
                        help='hidden dimension of the predictor (default: 512)')

    opt = parser.parse_args()

    # Set model and tensorboard paths
    opt.model_path = '/model'
    opt.tb_path = '/model'

    # Adjust crop size for ImageNet
    if opt.dataset == 'imagenet' and 'alexnet' not in opt.model:
        opt.crop = 0.08

    # Parse learning rate decay epochs
    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = [int(it) for it in iterations]

    # Set loss method
    opt.method = 'softmax' if opt.softmax else 'nce'

    print(opt)
    print(opt.model_name)

    # Create model and tensorboard directories
    opt.model_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.model_folder):
        os.makedirs(opt.model_folder)

    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)

    print("Save model to:", opt.model_folder)
    return opt


def moment_update(model, model_ema, m):
    """
    Update model_ema parameters using exponential moving average.

    Args:
        model: Current model
        model_ema: Exponential moving average model
        m: Momentum coefficient
    """
    for p1, p2 in zip(model.parameters(), model_ema.parameters()):
        p2.data.mul_(m).add_(1 - m, p1.detach().data)


def get_shuffle_ids(bsz):
    """
    Generate shuffle indices for ShuffleBN.

    Args:
        bsz: Batch size

    Returns:
        forward_inds: Forward shuffle indices
        backward_inds: Backward shuffle indices
    """
    forward_inds = torch.randperm(bsz).long().cuda()
    backward_inds = torch.zeros(bsz).long().cuda()
    value = torch.arange(bsz).long().cuda()
    backward_inds.index_copy_(0, forward_inds, value)
    return forward_inds, backward_inds


def main():
    """Main training function."""
    args = parse_option()

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    # Set data folder path
    if args.dataset == 'imagenet100':
        data_folder = os.path.join(args.data_folder, 'train')
    else:
        data_folder = args.data_folder

    # Define image transformations
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
    # Create model and optimizer
    n_data = len(train_dataset)

    if args.model == 'resnet50':
        encoder = resnet50(num_classes=args.n_way, zero_init_residual=True)
        if args.moco:
            model_ema = resnet50()
    else:
        raise NotImplementedError('Model not supported: {}'.format(args.model))

    model = SimSiam(encoder, args.dim, args.pred_dim)
    encoder = model.encoder
    model = torch.nn.DataParallel(model)
    encoder = torch.nn.DataParallel(encoder)

    # Copy weights from model to model_ema
    if args.moco:
        moment_update(model, model_ema, 0)

    # Set up contrast memory and criterion
    if args.moco:
        contrast = MemoryMoCo(
            128, n_data, args.nce_k, args.nce_t, args.softmax
        ).cuda(args.gpu)
    else:
        contrast = MemoryInsDis(
            128, n_data, args.nce_k, args.nce_t, args.nce_m, args.softmax
        ).cuda(args.gpu)

    criterion = torch.nn.CrossEntropyLoss().cuda(args.gpu)
    criterion_simsiam = torch.nn.CosineSimilarity(dim=1).cuda(args.gpu)

    model = model.cuda()

    # Set up optimizer with different learning rates for encoder and predictor
    optim_params = [
        {'params': model.module.encoder.parameters(), 'fix_lr': False},
        {'params': model.module.predictor.parameters(), 'fix_lr': True}
    ]

    optimizer = torch.optim.SGD(
        optim_params,
        lr=args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )

    cudnn.benchmark = True

    # Initialize mixed precision training if enabled
    if args.amp and amp is not None:
        model, optimizer = amp.initialize(
            model, optimizer, opt_level=args.opt_level
        )
        if args.moco:
            optimizer_ema = torch.optim.SGD(
                model_ema.parameters(), lr=0, momentum=0, weight_decay=0
            )
            model_ema, optimizer_ema = amp.initialize(
                model_ema, optimizer_ema, opt_level=args.opt_level
            )

    # Optionally resume from a checkpoint
    args.start_epoch = 1
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> Loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location='cpu')
            args.start_epoch = checkpoint['epoch'] + 1
            model.load_state_dict(checkpoint['model'])
            if args.moco:
                model_ema.load_state_dict(checkpoint['model_ema'])

            if args.amp and amp is not None and checkpoint['opt'].amp:
                print('==> Resuming amp state_dict')
                amp.load_state_dict(checkpoint['amp'])

            print("=> Loaded successfully '{}' (epoch {})".format(
                args.resume, checkpoint['epoch']))
            del checkpoint
            torch.cuda.empty_cache()
        else:
            print("=> No checkpoint found at '{}'".format(args.resume))

    # Training loop
    for epoch in range(args.start_epoch, args.epochs + 1):
        adjust_learning_rate(epoch, args, optimizer)
        print("==> Training...")

        time1 = time.time()
        loss, prob = train(
            epoch, train_loader, model, contrast, criterion,
            criterion_simsiam, optimizer, args
        )
        time2 = time.time()
        print('Epoch {}, total time {:.2f}s'.format(epoch, time2 - time1))

        # Save model checkpoint
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
            if args.amp and amp is not None:
                state['amp'] = amp.state_dict()
            save_file = os.path.join(
                args.model_folder, 'ckpt_epoch_{}.pth'.format(epoch)
            )
            torch.save(state, save_file)
            print("Saved model to:", save_file)
            # Release GPU memory
            del state
            torch.cuda.empty_cache()


def train(epoch, train_loader, model, contrast, criterion, criterion_simsiam,
          optimizer, opt):
    """
    Train for one epoch using instance discrimination.

    Args:
        epoch: Current epoch number
        train_loader: Training data loader
        model: Model to train
        contrast: Contrast memory module
        criterion: Loss criterion
        criterion_simsiam: SimSiam loss criterion
        optimizer: Optimizer
        opt: Training options

    Returns:
        Average loss and accuracy for the epoch
    """
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    prob_meter = AverageMeter()

    end = time.time()

    for idx, (ori_img, inputs, y) in enumerate(train_loader):
        data_time.update(time.time() - end)

        bsz = inputs.size(0)

        inputs = inputs.float()
        if opt.gpu is not None:
            inputs = inputs.cuda(opt.gpu, non_blocking=True)
        else:
            inputs = inputs.cuda()
        y = y.cuda()

        # Forward pass
        shuffle_ids, reverse_ids = get_shuffle_ids(bsz)

        _, output1, feature = model.encoder(inputs, notsimsiam=True)

        loss = criterion(output1, y.long())
        prob = (torch.eq(output1.argmax(dim=1), y).sum().item()) / y.shape[0]

        # Backward pass
        optimizer.zero_grad()
        if opt.amp and amp is not None:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        optimizer.step()

        # Update meters
        loss_meter.update(loss.item(), bsz)
        prob_meter.update(prob, bsz)

        torch.cuda.synchronize()
        batch_time.update(time.time() - end)
        end = time.time()

        # Print training info
        if (idx + 1) % opt.print_freq == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})\t'
                  'prob {prob.val:.3f} ({prob.avg:.3f})'.format(
                      epoch, idx + 1, len(train_loader),
                      batch_time=batch_time, data_time=data_time,
                      loss=loss_meter, prob=prob_meter))
            sys.stdout.flush()
        # Anti-CAM training (after epoch threshold)
        if epoch >= opt.epoch_t:
            antiCAMs = getCAM(inputs, output1, feature, model.encoder, opt)

            # Anti-CAM forward pass
            p1, p2, z1, z2, _, anti_output = model(x1=inputs, x2=antiCAMs)
            sim_loss = -(criterion_simsiam(p1, z2).mean() +
                         criterion_simsiam(p2, z1).mean()) * 0.5
            anti_loss = criterion(anti_output, y.long())
            loss = anti_loss + sim_loss
            prob = (torch.eq(anti_output.argmax(dim=1), y).sum().item()) / y.shape[0]

            # Backward pass
            optimizer.zero_grad()
            if opt.amp and amp is not None:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            optimizer.step()

            # Update meters
            loss_meter.update(loss.item(), bsz)
            prob_meter.update(prob, bsz)

            torch.cuda.synchronize()
            batch_time.update(time.time() - end)
            end = time.time()

            # Print training info
            if (idx + 1) % opt.print_freq == 0:
                print('Train: [{0}][{1}/{2}]\t'
                      'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'loss {loss.val:.3f} ({loss.avg:.3f})\t'
                      'prob {prob.val:.3f} ({prob.avg:.3f})'.format(
                          epoch, idx + 1, len(train_loader),
                          batch_time=batch_time, data_time=data_time,
                          loss=loss_meter, prob=prob_meter))
                sys.stdout.flush()

    return loss_meter.avg, prob_meter.avg


def getCAM(inputs, output1, feature, model, opt, train_transform=None):
    """
    Generate Class Activation Maps (CAM) for inputs.

    Args:
        inputs: Input images
        output1: Model outputs
        feature: Feature maps
        model: Model
        opt: Options
        train_transform: Optional transform

    Returns:
        Anti-CAM processed images
    """
    logits = output1.detach()  # [bs, n_way]
    features_blobs = feature.detach()
    params = list(model.parameters())[-2]
    weight_softmax = params.detach().data.squeeze()  # [n_way, nce_k]

    if opt.dataset in ['miniimagenet', 'tieredimage', 'imagenet100']:
        size_upsample = (84, 84)

    for i, logit in enumerate(logits):
        h_x = torch.nn.functional.softmax(logit, dim=0).data
        probs, idx_cam = h_x.sort(0, True)

        CAM = returnCAM_tensor(
            features_blobs[i], weight_softmax, [idx_cam[0]],
            size_upsample=size_upsample, use_gpu=True
        )
        anti_cam = antiCamFunction(CAM, mode=opt.cam_mode, t=opt.cam_t)
        if i == 0:
            antiCAMs = torch.mul(anti_cam, inputs[i])
        else:
            antiCAMs = torch.cat([
                antiCAMs, torch.mul(anti_cam, inputs[i])
            ], dim=0)
    return antiCAMs


def antiCamFunction(CAM, mode='reverse', t=None):
    """
    Apply anti-CAM function to reverse attention.

    Args:
        CAM: Class activation map
        mode: Processing mode
        t: Threshold (unused in reverse mode)

    Returns:
        Reversed CAM
    """
    if mode == 'reverse':
        return 1 - CAM[0].unsqueeze(0)


def rand_bbox(size, lam):
    """
    Generate random bounding box for CutMix.

    Args:
        size: Image size
        lam: Lambda parameter

    Returns:
        Bounding box coordinates (bbx1, bby1, bbx2, bby2)
    """
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # Uniform sampling
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def uniform_loss(x, t=2):
    """
    Calculate uniform loss for feature distribution.

    Args:
        x: Features
        t: Temperature parameter

    Returns:
        Uniform loss value
    """
    return torch.pdist(x, p=2).pow(2).mul(-t).exp().mean().log()


if __name__ == '__main__':
    main()
