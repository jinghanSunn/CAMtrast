"""
Noise Contrastive Estimation (NCE) loss functions.

Implements various contrastive learning loss functions including NCE,
InfoNCE, and knowledge distillation losses.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

eps = 1e-7


class NCECriterion(nn.Module):
    """
    Noise Contrastive Estimation loss.

    Implements the NCE loss from Eq. (12) in the original NCE paper.
    """

    def __init__(self, n_data):
        """
        Initialize NCE criterion.

        Args:
            n_data: Number of data samples
        """
        super(NCECriterion, self).__init__()
        self.n_data = n_data

    def forward(self, x):
        """
        Compute NCE loss.

        Args:
            x: Input tensor of shape (batch_size, K+1) containing
               positive and negative similarities

        Returns:
            NCE loss value
        """
        bsz = x.shape[0]
        m = x.size(1) - 1

        # Noise distribution
        Pn = 1 / float(self.n_data)

        # Loss for positive pair
        P_pos = x.select(1, 0)
        log_D1 = torch.div(P_pos, P_pos.add(m * Pn + eps)).log_()

        # Loss for K negative pairs
        P_neg = x.narrow(1, 1, m)
        log_D0 = torch.div(
            P_neg.clone().fill_(m * Pn), P_neg.add(m * Pn + eps)
        ).log_()

        loss = -(log_D1.sum(0) + log_D0.view(-1, 1).sum(0)) / bsz

        return loss


class NCESoftmaxLoss(nn.Module):
    """
    InfoNCE loss (softmax cross-entropy).

    Also known as info-NCE loss in the CPC paper.
    """

    def __init__(self):
        """Initialize InfoNCE loss."""
        super(NCESoftmaxLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        """
        Compute InfoNCE loss.

        Args:
            x: Input tensor of similarities

        Returns:
            InfoNCE loss value
        """
        bsz = x.shape[0]
        x = x.squeeze()
        label = torch.zeros([bsz]).cuda().long()
        loss = self.criterion(x, label)
        return loss


class DistillKL(nn.Module):
    """KL divergence loss for knowledge distillation."""

    def __init__(self, T):
        """
        Initialize KL divergence loss.

        Args:
            T: Temperature for softmax
        """
        super(DistillKL, self).__init__()
        self.T = T

    def forward(self, y_s, y_t):
        """
        Compute KL divergence between student and teacher.

        Args:
            y_s: Student logits
            y_t: Teacher logits

        Returns:
            KL divergence loss
        """
        p_s = F.log_softmax(y_s / self.T, dim=1)
        p_t = F.softmax(y_t / self.T, dim=1)
        loss = F.kl_div(p_s, p_t, reduction='sum') * (self.T ** 2) / y_s.shape[0]
        return loss


def CrossEntropyLoss_label_smooth(targets, num_classes=10, epsilon=0.1):
    """
    Create label-smoothed targets for cross-entropy loss.

    Args:
        targets: Target labels
        num_classes: Number of classes
        epsilon: Smoothing parameter

    Returns:
        Smoothed label tensor
    """
    N = len(targets)
    targets = torch.Tensor(targets).long().unsqueeze(1)

    # Initialize matrix with epsilon / (num_classes - 1)
    smoothed_labels = torch.full(
        size=(N, num_classes),
        fill_value=epsilon / (num_classes - 1)
    )

    # Set target positions to 1 - epsilon
    smoothed_labels.scatter_(dim=1, index=targets, value=1 - epsilon)

    return smoothed_labels
