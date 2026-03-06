"""
Utility functions for training and evaluation.

Provides helper functions for learning rate scheduling, metrics calculation,
data augmentation, and CAM generation.
"""
from __future__ import print_function

import cv2
import numpy as np
import torch
from torchvision import transforms

def adjust_learning_rate(epoch, opt, optimizer):
    """
    Adjust learning rate based on epoch and decay schedule.

    Args:
        epoch: Current epoch number
        opt: Options containing lr_decay_epochs and lr_decay_rate
        optimizer: Optimizer to update
    """
    steps = np.sum(epoch > np.asarray(opt.lr_decay_epochs))
    if steps > 0:
        new_lr = opt.learning_rate * (opt.lr_decay_rate ** steps)
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr


class AverageMeter(object):
    """Computes and stores the average and current value."""

    def __init__(self):
        """Initialize meter."""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.reset()

    def reset(self):
        """Reset all statistics."""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """
        Update statistics with new value.

        Args:
            val: New value
            n: Number of samples (default: 1)
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """
    Compute accuracy over the k top predictions.

    Args:
        output: Model output logits
        target: Ground truth labels
        topk: Tuple of k values for top-k accuracy

    Returns:
        List of top-k accuracies
    """
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
    """
    Validate model on validation set.

    Args:
        model: Model to validate
        device: Device to use
        val_loader: Validation data loader
        criterion: Loss criterion

    Returns:
        Tuple of (error_rate, average_loss, margin)
    """
    sum_loss, sum_correct = 0, 0
    margin = torch.Tensor([]).to(device)

    # Switch to evaluation mode
    model.eval()
    with torch.no_grad():
        for i, (data, target) in enumerate(val_loader):
            data, target = data.to(device), target.to(device)

            # Compute output
            output = model(data)

            # Compute classification error and loss
            pred = output.max(1)[1]
            sum_correct += pred.eq(target).sum().item()
            sum_loss += len(data) * criterion(output, target).item()

            # Compute margin
            output_m = output.clone()
            for i in range(target.size(0)):
                output_m[i, target[i]] = output_m[i, :].min()
            margin = torch.cat((
                margin,
                output[:, target].diag() - output_m[:, output_m.max(1)[1]].diag()
            ), 0)

        val_margin = np.percentile(margin.cpu().numpy(), 10)

    error_rate = 1 - (sum_correct / len(val_loader.dataset))
    avg_loss = sum_loss / len(val_loader.dataset)
    return error_rate, avg_loss, val_margin

def mixup_data(x, y, alpha, num_classes):
    """
    Compute mixup data augmentation.

    Args:
        x: Input data
        y: Labels
        alpha: Mixup alpha parameter
        num_classes: Number of classes

    Returns:
        Tuple of (mixed_inputs, mixed_targets, lambda)
    """
    if alpha > 0.:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).cuda()
    mixed_x = lam * x + (1 - lam) * x[index, :]

    y_a = torch.nn.functional.one_hot(y, num_classes=num_classes)
    y_b = torch.nn.functional.one_hot(y[index], num_classes=num_classes)
    y_mix = lam * y_a + (1 - lam) * y_b

    return mixed_x, y_mix, lam


def returnCAM_tensor(feature_conv, weight_softmax, class_idx,
                     size_upsample=(84, 84), use_gpu=False):
    """
    Generate Class Activation Maps (CAM) from feature maps.

    Args:
        feature_conv: Convolutional feature maps
        weight_softmax: Softmax layer weights
        class_idx: Class indices to generate CAM for
        size_upsample: Target size for upsampling
        use_gpu: Whether to use GPU

    Returns:
        Upsampled CAM tensor
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    nc, h, w = feature_conv.shape

    for idx in class_idx:
        cam = torch.mm(
            weight_softmax[idx].reshape(1, weight_softmax[idx].shape[0]),
            feature_conv.reshape((nc, h * w))
        ).data
        cam = cam.reshape(h, w)
        cam = cam - torch.min(cam)
        cam_img = cam / torch.max(cam)
        cam_img = (255 * cam_img.cpu().numpy())
        output_cam = (transform(cv2.resize(cam_img, size_upsample)) / 255.0).unsqueeze(0)

    return output_cam.cuda()


def confidence_interval(std, n):
    """
    Calculate 95% confidence interval.

    Args:
        std: Standard deviation
        n: Number of samples

    Returns:
        Confidence interval value
    """
    return 1.96 * std / np.sqrt(n)


if __name__ == '__main__':
    meter = AverageMeter()
                                                                              