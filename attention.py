"""
This script defines two attention modules: Positional Attention Module (PAM_Module) and Channel Attention Module (CAM_Module).
These modules are designed to be used in deep learning models, particularly for image processing tasks.
"""
import torch
from torch.nn import Module,Conv2d,Parameter,Softmax

torch_ver = torch.__version__[:3]

__all__ = ["PAM_Module", "CAM_Module"]


class PAM_Module(Module):
    """
    Positional Attention Module (PAM_Module)
    
    This module is designed to capture spatial relationships between different positions in the input feature maps.
    It uses a self-attention mechanism to compute attention weights based on spatial positions.
    
    Attributes:
        chanel_in (int): Number of input channels.
        query_conv (Conv2d): Convolutional layer to compute query features.
        key_conv (Conv2d): Convolutional layer to compute key features.
        value_conv (Conv2d): Convolutional layer to compute value features.
        gamma (Parameter): Learnable parameter to scale the attention output.
        softmax (Softmax): Softmax function to normalize attention weights.
    """

    def __init__(self, in_dim):
        super(PAM_Module, self).__init__()
        self.chanel_in = in_dim
        self.query_conv = Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = Parameter(torch.zeros(1))

        self.softmax = Softmax(dim=-1)

    def forward(self, x):
        """
        inputs :
            x : input feature maps(B x C x H x W)
        returns :
            out : attention value + input feature
            attention: B x (H,W) x (H,W)
        """
        x = x.squeeze(-1)
        m_batchsize, C, height, width = x.size()
        proj_query = (
            self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        )
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)

        out = (self.gamma * out + x).unsqueeze(-1)
        return out


class CAM_Module(Module):
    """
    Channel Attention Module (CAM_Module)
    
    This module is designed to capture channel-wise relationships in the input feature maps.
    It uses a self-attention mechanism to compute attention weights based on channel interactions.
    
    Attributes:
        chanel_in (int): Number of input channels.
        gamma (Parameter): Learnable parameter to scale the attention output.
        softmax (Softmax): Softmax function to normalize attention weights.
    """

    def __init__(self, in_dim):
        super(CAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.gamma = Parameter(torch.zeros(1))
        self.softmax = Softmax(dim=-1)

    def forward(self, x):
        """
        inputs :
            x : input feature maps(B x C x H x W)
        returns :
            out : attention value + input feature
            attention: B x C x C
        """
        m_batchsize, C, height, width, channle = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width, channle)
        out = self.gamma * out + x
        return out
