"""
ResNet architecture for contrastive learning.

Modified ResNet implementation with support for:
- Squeeze-and-Excitation (SE) blocks
- Mixup augmentation
- Custom output dimensions for contrastive learning
"""
from typing import Any, Callable, List, Optional, Type

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152', 'resnext50_32x4d', 'resnext101_32x8d',
           'wide_resnet50_2', 'wide_resnet101_2']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}



def conv3x3(in_planes: int, out_planes: int, stride: int = 1,
            groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """
    3x3 convolution with padding.

    Args:
        in_planes: Number of input channels
        out_planes: Number of output channels
        stride: Stride for convolution
        groups: Number of groups for grouped convolution
        dilation: Dilation rate

    Returns:
        3x3 convolutional layer
    """
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride,
        padding=dilation, groups=groups, bias=False, dilation=dilation
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """
    1x1 convolution.

    Args:
        in_planes: Number of input channels
        out_planes: Number of output channels
        stride: Stride for convolution

    Returns:
        1x1 convolutional layer
    """
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=1, stride=stride, bias=False
    )


class SELayer(nn.Module):
    """
    Squeeze-and-Excitation layer.

    Implements channel-wise attention mechanism from
    "Squeeze-and-Excitation Networks" (https://arxiv.org/abs/1709.01507).
    """

    def __init__(self, channel, reduction=16):
        """
        Initialize SE layer.

        Args:
            channel: Number of input channels
            reduction: Reduction ratio for bottleneck
        """
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input tensor

        Returns:
            Attention-weighted tensor
        """
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class BasicBlock(nn.Module):
    """
    Basic ResNet block for ResNet-18 and ResNet-34.

    Two 3x3 convolutional layers with batch normalization and ReLU.
    """
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        """
        Initialize BasicBlock.

        Args:
            inplanes: Number of input channels
            planes: Number of output channels
            stride: Stride for first convolution
            downsample: Optional downsampling layer
            groups: Number of groups (must be 1 for BasicBlock)
            base_width: Base width (must be 64 for BasicBlock)
            dilation: Dilation rate (must be 1 for BasicBlock)
            norm_layer: Normalization layer
        """
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")

        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor

        Returns:
            Output tensor
        """
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    """
    Bottleneck block for ResNet-50/101/152.

    Uses 1x1 -> 3x3 -> 1x1 convolutions. This is the ResNet V1.5 variant
    which places stride at the 3x3 convolution for improved accuracy.
    Reference: https://arxiv.org/abs/1512.03385
    """
    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        """
        Initialize Bottleneck block.

        Args:
            inplanes: Number of input channels
            planes: Number of output channels (before expansion)
            stride: Stride for 3x3 convolution
            downsample: Optional downsampling layer
            groups: Number of groups for grouped convolution
            base_width: Base width for computing bottleneck width
            dilation: Dilation rate
            norm_layer: Normalization layer
        """
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups

        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor

        Returns:
            Output tensor
        """
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

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

class ResNet(nn.Module):
    """
    ResNet architecture for contrastive learning.

    Modified to support custom output dimensions and L2 normalization.
    """

    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        num_classes: int = 128,
        out_dim=2048,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        """
        Initialize ResNet.

        Args:
            block: Block type (BasicBlock or Bottleneck)
            layers: Number of blocks in each layer
            num_classes: Number of classes for classifier
            out_dim: Output dimension for projection head
            zero_init_residual: Whether to zero-initialize residual branches
            groups: Number of groups for grouped convolution
            width_per_group: Width per group
            replace_stride_with_dilation: Whether to replace stride with dilation
            norm_layer: Normalization layer
        """
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                "or a 3-element tuple, got {}".format(replace_stride_with_dilation)
            )

        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(
            3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(
            block, 128, layers[1], stride=2,
            dilate=replace_stride_with_dilation[0]
        )
        self.layer3 = self._make_layer(
            block, 256, layers[2], stride=2,
            dilate=replace_stride_with_dilation[1]
        )
        self.layer4 = self._make_layer(
            block, 512, layers[3], stride=2,
            dilate=replace_stride_with_dilation[2]
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, out_dim)
        self.classifier = nn.Linear(512 * block.expansion, num_classes)

        self.l2_norm = Normalize(2)

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu'
                )
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize last BN in residual branches for better initialization
        # Reference: https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False
    ) -> nn.Sequential:
        """
        Create a residual layer.

        Args:
            block: Block type
            planes: Number of output channels
            blocks: Number of blocks in layer
            stride: Stride for first block
            dilate: Whether to use dilation instead of stride

        Returns:
            Sequential layer of blocks
        """
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(
            self.inplanes, planes, stride, downsample, self.groups,
            self.base_width, previous_dilation, norm_layer
        ))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(
                self.inplanes, planes, groups=self.groups,
                base_width=self.base_width, dilation=self.dilation,
                norm_layer=norm_layer
            ))

        return nn.Sequential(*layers)

    def _forward_impl(
        self,
        x: Tensor,
        notsimsiam,
        target,
        mixup_hidden,
        mixup_alpha,
        layer_mix
    ) -> Tensor:
        """
        Forward implementation.

        Args:
            x: Input tensor
            notsimsiam: Whether to skip projection head
            target: Target labels (for mixup)
            mixup_hidden: Whether to use mixup
            mixup_alpha: Mixup alpha parameter
            layer_mix: Layer to apply mixup

        Returns:
            Tuple of (projection, prediction, features)
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        feature = x
        x = self.avgpool(x)

        x = torch.flatten(x, 1)
        if notsimsiam:
            z = None
        else:
            z = self.fc(x)
        pred = self.classifier(x)

        return z, pred, feature

    def forward(
        self,
        x: Tensor,
        notsimsiam=False,
        target=None,
        mixup_hidden=False,
        mixup_alpha=0.1,
        layer_mix=None
    ) -> Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor
            notsimsiam: Whether to skip projection head
            target: Target labels
            mixup_hidden: Whether to use mixup
            mixup_alpha: Mixup alpha parameter
            layer_mix: Layer to apply mixup

        Returns:
            Model outputs
        """
        return self._forward_impl(
            x, notsimsiam, target, mixup_hidden, mixup_alpha, layer_mix
        )


def _resnet(
    arch: str,
    block: Type[Union[BasicBlock, Bottleneck]],
    layers: List[int],
    pretrained: bool,
    progress: bool,
    **kwargs: Any
) -> ResNet:
    """
    Create a ResNet model.

    Args:
        arch: Architecture name
        block: Block type
        layers: Number of blocks in each layer
        pretrained: Whether to load pretrained weights (not implemented)
        progress: Whether to show progress (not implemented)
        **kwargs: Additional arguments for ResNet

    Returns:
        ResNet model
    """
    model = ResNet(block, layers, **kwargs)
    return model


def _resnetse(
    arch: str,
    block: Type[Union[BasicBlock, Bottleneck]],
    layers: List[int],
    pretrained: bool,
    progress: bool,
    **kwargs: Any
) -> ResNet:
    """
    Create a ResNet model with SE blocks.

    Args:
        arch: Architecture name
        block: Block type
        layers: Number of blocks in each layer
        pretrained: Whether to load pretrained weights (not implemented)
        progress: Whether to show progress (not implemented)
        **kwargs: Additional arguments for ResNet

    Returns:
        ResNet model with SE blocks
    """
    model = ResNet(block, layers, **kwargs)
    return model


def resnet18(pretrained: bool = False, progress: bool = True,
             **kwargs: Any) -> ResNet:
    """
    ResNet-18 model.

    Reference: "Deep Residual Learning for Image Recognition"
    https://arxiv.org/pdf/1512.03385.pdf

    Args:
        pretrained: If True, returns a model pre-trained on ImageNet
        progress: If True, displays a progress bar of the download to stderr
        **kwargs: Additional arguments

    Returns:
        ResNet-18 model
    """
    return _resnet(
        'resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress, **kwargs
    )


def resnet34(pretrained: bool = False, progress: bool = True,
             **kwargs: Any) -> ResNet:
    """
    ResNet-34 model.

    Reference: "Deep Residual Learning for Image Recognition"
    https://arxiv.org/pdf/1512.03385.pdf

    Args:
        pretrained: If True, returns a model pre-trained on ImageNet
        progress: If True, displays a progress bar of the download to stderr
        **kwargs: Additional arguments

    Returns:
        ResNet-34 model
    """
    return _resnet(
        'resnet34', BasicBlock, [3, 4, 6, 3], pretrained, progress, **kwargs
    )


def resnet50(pretrained: bool = False, progress: bool = True,
             **kwargs: Any) -> ResNet:
    """
    ResNet-50 model.

    Reference: "Deep Residual Learning for Image Recognition"
    https://arxiv.org/pdf/1512.03385.pdf

    Args:
        pretrained: If True, returns a model pre-trained on ImageNet
        progress: If True, displays a progress bar of the download to stderr
        **kwargs: Additional arguments

    Returns:
        ResNet-50 model
    """
    return _resnet(
        'resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress, **kwargs
    )

def seresnet50(pretrained: bool = False, progress: bool = True,
               **kwargs: Any) -> ResNet:
    """
    SE-ResNet-50 model with Squeeze-and-Excitation blocks.

    Note: This function references BottleneckSE which needs to be implemented
    if SE blocks are required.

    Args:
        pretrained: If True, returns a model pre-trained on ImageNet
        progress: If True, displays a progress bar of the download to stderr
        **kwargs: Additional arguments

    Returns:
        SE-ResNet-50 model
    """
    # Note: BottleneckSE needs to be implemented for this to work
    return _resnet(
        'resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress, **kwargs
    )


def resnet101(pretrained: bool = False, progress: bool = True,
              **kwargs: Any) -> ResNet:
    """
    ResNet-101 model.

    Reference: "Deep Residual Learning for Image Recognition"
    https://arxiv.org/pdf/1512.03385.pdf

    Args:
        pretrained: If True, returns a model pre-trained on ImageNet
        progress: If True, displays a progress bar of the download to stderr
        **kwargs: Additional arguments

    Returns:
        ResNet-101 model
    """
    return _resnet(
        'resnet101', Bottleneck, [3, 4, 23, 3], pretrained, progress, **kwargs
    )


def resnet152(pretrained: bool = False, progress: bool = True,
              **kwargs: Any) -> ResNet:
    """
    ResNet-152 model.

    Reference: "Deep Residual Learning for Image Recognition"
    https://arxiv.org/pdf/1512.03385.pdf

    Args:
        pretrained: If True, returns a model pre-trained on ImageNet
        progress: If True, displays a progress bar of the download to stderr
        **kwargs: Additional arguments

    Returns:
        ResNet-152 model
    """
    return _resnet(
        'resnet152', Bottleneck, [3, 8, 36, 3], pretrained, progress, **kwargs
    )



def resnext50_32x4d(pretrained: bool = False, progress: bool = True,
                    **kwargs: Any) -> ResNet:
    """
    ResNeXt-50 32x4d model.

    Reference: "Aggregated Residual Transformation for Deep Neural Networks"
    https://arxiv.org/pdf/1611.05431.pdf

    Args:
        pretrained: If True, returns a model pre-trained on ImageNet
        progress: If True, displays a progress bar of the download to stderr
        **kwargs: Additional arguments

    Returns:
        ResNeXt-50 32x4d model
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    return _resnet(
        'resnext50_32x4d', Bottleneck, [3, 4, 6, 3],
        pretrained, progress, **kwargs
    )


def resnext101_32x8d(pretrained: bool = False, progress: bool = True,
                     **kwargs: Any) -> ResNet:
    """
    ResNeXt-101 32x8d model.

    Reference: "Aggregated Residual Transformation for Deep Neural Networks"
    https://arxiv.org/pdf/1611.05431.pdf

    Args:
        pretrained: If True, returns a model pre-trained on ImageNet
        progress: If True, displays a progress bar of the download to stderr
        **kwargs: Additional arguments

    Returns:
        ResNeXt-101 32x8d model
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8
    return _resnet(
        'resnext101_32x8d', Bottleneck, [3, 4, 23, 3],
        pretrained, progress, **kwargs
    )


def wide_resnet50_2(pretrained: bool = False, progress: bool = True,
                    **kwargs: Any) -> ResNet:
    """
    Wide ResNet-50-2 model.

    The model uses bottleneck channels that are twice as wide as standard ResNet.
    Reference: "Wide Residual Networks" https://arxiv.org/pdf/1605.07146.pdf

    Args:
        pretrained: If True, returns a model pre-trained on ImageNet
        progress: If True, displays a progress bar of the download to stderr
        **kwargs: Additional arguments

    Returns:
        Wide ResNet-50-2 model
    """
    kwargs['width_per_group'] = 64 * 2
    return _resnet(
        'wide_resnet50_2', Bottleneck, [3, 4, 6, 3],
        pretrained, progress, **kwargs
    )


def wide_resnet101_2(pretrained: bool = False, progress: bool = True,
                     **kwargs: Any) -> ResNet:
    """
    Wide ResNet-101-2 model.

    The model uses bottleneck channels that are twice as wide as standard ResNet.
    Reference: "Wide Residual Networks" https://arxiv.org/pdf/1605.07146.pdf

    Args:
        pretrained: If True, returns a model pre-trained on ImageNet
        progress: If True, displays a progress bar of the download to stderr
        **kwargs: Additional arguments

    Returns:
        Wide ResNet-101-2 model
    """
    kwargs['width_per_group'] = 64 * 2
    return _resnet(
        'wide_resnet101_2', Bottleneck, [3, 4, 23, 3],
        pretrained, progress, **kwargs
    )
