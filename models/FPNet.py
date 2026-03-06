"""Feature Projection Network for domain adaptation."""
import torch
import torch.nn as nn
from torch.autograd import Function


class FPNet(nn.Module):
    """
    Feature Projection Network.

    Separates features into domain-specific and domain-invariant components.
    """

    def __init__(self, in_channels, num_class):
        """
        Initialize FPNet.

        Args:
            in_channels: Number of input channels
            num_class: Number of output classes
        """
        super(FPNet, self).__init__()
        out_channels = 128
        self.FC1 = nn.Linear(in_channels, in_channels)
        self.FC2 = nn.Linear(in_channels, out_channels)
        self.classifier_fc = nn.Linear(out_channels, num_class)

        self.FP1 = nn.Linear(in_channels, in_channels)
        self.FP2 = nn.Linear(in_channels, out_channels)
        self.classifier_fp = nn.Linear(out_channels, num_class)

        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input features

        Returns:
            Tuple of (domain classifier output, feature projection output)
        """
        fp = self.FC1(x)
        fp = self.relu(fp)
        fp = self.FC2(fp)
        fp = self.relu(fp)

        fc = self.FC1(x)
        fc = self.relu(fc)
        fc = self.FC2(fc)
        fc = self.relu(fc)

        fc = GradReverse.apply(fc, 1.0)

        out_c = self.classifier_fc(fc)

        fp_ = NB_algorithm(fp, fc)

        self.bn(fp_)
        out_p = self.classifier_fp(fp_)

        return out_c, out_p

    def proj(self, x, y):
        """
        Project x onto y.

        Args:
            x: Vector to project
            y: Vector to project onto

        Returns:
            Projection of x onto y
        """
        return (x * y / torch.norm(y, p=2)) * (y / torch.norm(y, p=2))


class GradReverse(Function):
    """Gradient reversal layer for domain adaptation."""

    @staticmethod
    def forward(ctx, x, lambd, **kwargs: None):
        """
        Forward pass (identity function).

        Args:
            ctx: Context object
            x: Input tensor
            lambd: Lambda parameter for gradient reversal
            **kwargs: Additional arguments

        Returns:
            Input tensor unchanged
        """
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass (reverse gradient).

        Args:
            ctx: Context object
            grad_output: Gradient from next layer

        Returns:
            Reversed gradient
        """
        return grad_output * -ctx.lambd, None

class Matrix(object):
    """Helper class for vector operations in feature space."""

    def __init__(self, coordinates):
        """
        Initialize Matrix.

        Args:
            coordinates: Tensor of coordinates
        """
        self.coordinates = coordinates
        self.num_samples = coordinates.shape[0]
        self.dimension = coordinates.shape[1]

    def __str__(self):
        """String representation."""
        return str(self.coordinates)

    def plus(self, v):
        """Add two vectors."""
        return self.coordinates + v.coordinates

    def minus(self, v):
        """Subtract two vectors."""
        return self.coordinates - v.coordinates

    def magnitude(self):
        """Compute L2 norm of vectors."""
        return torch.sum(
            (self.coordinates).pow(2), dim=-1
        ).reshape((self.num_samples, 1))

    def normalized(self):
        """Normalize vectors to unit length."""
        magnitude = self.magnitude()
        weight = (1.0 / magnitude).reshape(self.num_samples, 1)
        return self.coordinates * weight

    def component_parallel_to(self, basis):
        """Compute component parallel to basis."""
        u = basis.normalized()
        weight = torch.sum(
            self.coordinates * u, dim=-1
        ).reshape(self.num_samples, 1)
        return u * weight

    def component_orthogonal_to(self, basis):
        """Compute component orthogonal to basis."""
        projection = self.component_parallel_to(basis)
        return self.coordinates - projection


def NB_algorithm(original_feature, trivial_feature):
    """
    Null-space Based algorithm for feature projection.

    Removes trivial features from original features using orthogonal projection.

    Args:
        original_feature: Original feature tensor
        trivial_feature: Trivial/domain-specific feature tensor

    Returns:
        Projected features with trivial components removed
    """
    original_feature = Matrix(original_feature)
    trivial_feature = Matrix(trivial_feature)
    d = original_feature.component_orthogonal_to(trivial_feature)
    d = Matrix(d)
    f = original_feature.component_parallel_to(d)
    return f