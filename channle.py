"""
This module implements a basic building block for a convolutional neural network (CNN) with channel and spatial attention mechanisms.
The attention mechanisms are designed to enhance the representation power of the network by selectively focusing on important features.
"""

import torch
import torch.nn as nn


def conv3x3(in_planes, out_planes, stride=1):
    """
    Creates a 3x3 convolutional layer with padding to maintain spatial dimensions.

    Args:
        in_planes (int): Number of input channels.
        out_planes (int): Number of output channels.
        stride (int, optional): Stride of the convolution. Defaults to 1.

    Returns:
        nn.Conv2d: A 3x3 convolutional layer with padding.
    """
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
    )


class ChannelAttention(nn.Module):
    """
    Channel Attention Module (CAM).

    This module computes channel-wise attention weights using average and max pooling,
    followed by a shared fully connected layer and a sigmoid activation function.
    The attention weights are then multiplied with the input feature maps to enhance important channels.

    Attributes:
        avg_pool (nn.AdaptiveAvgPool2d): Global average pooling layer.
        max_pool (nn.AdaptiveMaxPool2d): Global max pooling layer.
        fc1 (nn.Conv2d): First fully connected layer (shared).
        relu1 (nn.ReLU): ReLU activation function.
        fc2 (nn.Conv2d): Second fully connected layer.
        sigmoid (nn.Sigmoid): Sigmoid activation function.
    """

    def __init__(self, in_planes, ratio=16):
        """
        Initializes the ChannelAttention module.

        Args:
            in_planes (int): Number of input channels.
            ratio (int, optional): Reduction ratio for the fully connected layers. Defaults to 16.
        """
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Forward pass of the ChannelAttention module.

        Args:
            x (Tensor): Input feature maps with shape (B x C x H x W).

        Returns:
            Tensor: Channel attention weights with shape (B x C x 1 x 1).
        """
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    """
    Spatial Attention Module (SAM).

    This module computes spatial attention weights using average and max pooling,
    followed by a convolutional layer and a sigmoid activation function.
    The attention weights are then multiplied with the input feature maps to enhance important spatial regions.

    Attributes:
        conv1 (nn.Conv2d): Convolutional layer for spatial attention.
        sigmoid (nn.Sigmoid): Sigmoid activation function.
    """

    def __init__(self, kernel_size=7):
        """
        Initializes the SpatialAttention module.

        Args:
            kernel_size (int, optional): Size of the convolutional kernel. Defaults to 7.
        """
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Forward pass of the SpatialAttention module.

        Args:
            x (Tensor): Input feature maps with shape (B x C x H x W).

        Returns:
            Tensor: Spatial attention weights with shape (B x 1 x H x W).
        """
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class BasicBlock(nn.Module):
    """
    Basic building block for a convolutional neural network with channel and spatial attention.

    This block consists of two convolutional layers with batch normalization and ReLU activation,
    followed by channel and spatial attention modules. The attention weights are multiplied with
    the feature maps to enhance important channels and spatial regions.

    Attributes:
        expansion (int): Expansion factor for the block (default is 1).
        conv1 (nn.Conv2d): First convolutional layer.
        bn1 (nn.BatchNorm2d): Batch normalization layer for the first convolution.
        relu (nn.ReLU): ReLU activation function.
        conv2 (nn.Conv2d): Second convolutional layer.
        bn2 (nn.BatchNorm2d): Batch normalization layer for the second convolution.
        ca (ChannelAttention): Channel attention module.
        sa (SpatialAttention): Spatial attention module.
        downsample (nn.Module, optional): Downsample module for residual connection.
    """

    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        """
        Initializes the BasicBlock.

        Args:
            inplanes (int): Number of input channels.
            planes (int): Number of output channels.
            stride (int, optional): Stride of the convolution. Defaults to 1.
            downsample (nn.Module, optional): Downsample module for residual connection. Defaults to None.
        """
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.ca = ChannelAttention(planes)
        self.sa = SpatialAttention()

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        """
        Forward pass of the BasicBlock.

        Args:
            x (Tensor): Input feature maps with shape (B x C x H x W).

        Returns:
            Tensor: Output feature maps with shape (B x C x H x W).
        """
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = self.ca(out) * out  # Apply channel attention
        out = self.sa(out) * out  # Apply spatial attention

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out