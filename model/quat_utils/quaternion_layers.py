##########################################################
# pytorch-qnn v1.0
# Titouan Parcollet
# LIA, Université d'Avignon et des Pays du Vaucluse
# ORKIS, Aix-en-provence
# October 2018
##########################################################

import numpy as np
from numpy.random import RandomState
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn import Module
from .quaternion_ops import *
import math
import sys


class QuaternionTransposeConv(Module):
    """Applies a Quaternion Transposed Convolution (or Deconvolution) to the incoming data."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        dilation=1,
        padding=0,
        output_padding=0,
        groups=1,
        bias=True,
        init_criterion="he",
        weight_init="quaternion",
        seed=None,
        operation="convolution2d",
        rotation=False,
        quaternion_format=False,
    ):

        super(QuaternionTransposeConv, self).__init__()

        self.in_channels = in_channels // 4
        self.out_channels = out_channels // 4
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        self.dilation = dilation
        self.init_criterion = init_criterion
        self.weight_init = weight_init
        self.seed = seed if seed is not None else np.random.randint(0, 1234)
        self.rng = RandomState(self.seed)
        self.operation = operation
        self.rotation = rotation
        self.quaternion_format = quaternion_format
        self.winit = {
            "quaternion": quaternion_init,
            "unitary": unitary_init,
            "random": random_init,
        }[self.weight_init]

        (self.kernel_size, self.w_shape) = get_kernel_and_weight_shape(
            self.operation, self.out_channels, self.in_channels, kernel_size
        )

        self.r_weight = Parameter(torch.Tensor(*self.w_shape))
        self.i_weight = Parameter(torch.Tensor(*self.w_shape))
        self.j_weight = Parameter(torch.Tensor(*self.w_shape))
        self.k_weight = Parameter(torch.Tensor(*self.w_shape))

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        affect_init_conv(
            self.r_weight,
            self.i_weight,
            self.j_weight,
            self.k_weight,
            self.kernel_size,
            self.winit,
            self.rng,
            self.init_criterion,
        )
        if self.bias is not None:
            self.bias.data.zero_()

    def forward(self, input):
        return quaternion_transpose_conv(
            input,
            self.r_weight,
            self.i_weight,
            self.j_weight,
            self.k_weight,
            self.bias,
            self.stride,
            self.padding,
            self.output_padding,
            self.groups,
            self.dilation,
        )

    def __repr__(self):
        return (
            self.__class__.__name__
            + "("
            + "in_channels="
            + str(self.in_channels)
            + ", out_channels="
            + str(self.out_channels)
            + ", bias="
            + str(self.bias is not None)
            + ", kernel_size="
            + str(self.kernel_size)
            + ", stride="
            + str(self.stride)
            + ", padding="
            + str(self.padding)
            + ", dilation="
            + str(self.dilation)
            + ", init_criterion="
            + str(self.init_criterion)
            + ", weight_init="
            + str(self.weight_init)
            + ", seed="
            + str(self.seed)
            + ", operation="
            + str(self.operation)
            + ")"
        )


class QuaternionConv(Module):
    r"""Applies a Quaternion Convolution to the incoming data."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        dilation=1,
        padding=0,
        groups=1,
        bias=True,
        init_criterion="glorot",
        weight_init="quaternion",
        seed=None,
        operation="convolution2d",
        quaternion_format=True,
        scale=False,
    ):

        super(QuaternionConv, self).__init__()

        self.in_channels = in_channels // 4
        self.out_channels = out_channels // 4
        self.stride = stride
        self.padding = padding
        self.groups = groups
        self.dilation = dilation
        self.init_criterion = init_criterion
        self.weight_init = weight_init
        self.seed = seed if seed is not None else np.random.randint(0, 1234)
        self.rng = RandomState(self.seed)
        self.operation = operation
        self.quaternion_format = quaternion_format
        self.winit = {
            "quaternion": quaternion_init,
            "unitary": unitary_init,
            "random": random_init,
        }[self.weight_init]
        self.scale = scale

        (self.kernel_size, self.w_shape) = get_kernel_and_weight_shape(
            self.operation, self.in_channels, self.out_channels, kernel_size
        )

        # print("QuaternionConv kernel Shape: ", self.kernel_size)
        # print("QuaternionConv w Shape: ", self.w_shape)
        self.r_weight = Parameter(torch.Tensor(*self.w_shape))
        self.i_weight = Parameter(torch.Tensor(*self.w_shape))
        self.j_weight = Parameter(torch.Tensor(*self.w_shape))
        self.k_weight = Parameter(torch.Tensor(*self.w_shape))

        torch.nn.init.normal_(self.r_weight.data, std=0.02)
        torch.nn.init.normal_(self.i_weight.data, std=0.02)
        torch.nn.init.normal_(self.j_weight.data, std=0.02)
        torch.nn.init.normal_(self.k_weight.data, std=0.02)

        if self.scale:
            self.scale_param = Parameter(torch.Tensor(self.r_weight.shape))
        else:
            self.scale_param = None
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        affect_init_conv(
            self.r_weight,
            self.i_weight,
            self.j_weight,
            self.k_weight,
            self.kernel_size,
            self.winit,
            self.rng,
            self.init_criterion,
        )
        if self.scale_param is not None:
            torch.nn.init.xavier_uniform_(self.scale_param.data)
        if self.bias is not None:
            self.bias.data.zero_()

    def forward(self, input):
        return quaternion_conv(
            input,
            self.r_weight,
            self.i_weight,
            self.j_weight,
            self.k_weight,
            self.bias,
            self.stride,
            self.padding,
            self.groups,
            self.dilation,
        )

    def __repr__(self):
        return (
            self.__class__.__name__
            + "("
            + "in_channels="
            + str(self.in_channels)
            + ", out_channels="
            + str(self.out_channels)
            + ", bias="
            + str(self.bias is not None)
            + ", kernel_size="
            + str(self.kernel_size)
            + ", stride="
            + str(self.stride)
            + ", padding="
            + str(self.padding)
            + ", init_criterion="
            + str(self.init_criterion)
            + ", weight_init="
            + str(self.weight_init)
            + ", seed="
            + str(self.seed)
            + ", operation="
            + str(self.operation)
            + ")"
        )


class PixelShuffle2D(torch.nn.Module):
    """
    2D Pixel Shuffler
    Upscales height and width, downscales channel length
    "short" is input, "long" is output
    """

    def __init__(self, upscale_factor):
        super(PixelShuffle2D, self).__init__()
        self.upscale_factor = upscale_factor

    def forward(self, x):
        batch_size = x.shape[0]
        short_channel_len = x.shape[1]
        short_height = x.shape[2]
        short_width = x.shape[3]

        long_channel_len = short_channel_len // (self.upscale_factor ** 2)
        long_height = self.upscale_factor * short_height
        long_width = self.upscale_factor * short_width

        x = x.contiguous().view(
            [
                batch_size,
                self.upscale_factor,
                self.upscale_factor,
                long_channel_len,
                short_height,
                short_width,
            ]
        )
        x = x.permute(0, 3, 4, 1, 5, 2).contiguous()
        x = x.view(batch_size, long_channel_len, long_height, long_width)

        return x


class QuaternionLinearAutograd(Module):
    r"""Applies a quaternion linear transformation to the incoming data. A custom
    Autograd function is call to drastically reduce the VRAM consumption. Nonetheless, computing
    time is also slower compared to QuaternionLinear().
    """

    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        init_criterion="glorot",
        weight_init="quaternion",
        seed=None,
        rotation=False,
        quaternion_format=True,
        scale=False,
    ):

        super(QuaternionLinearAutograd, self).__init__()
        self.in_features = in_features // 4
        self.out_features = out_features // 4
        self.rotation = rotation
        self.quaternion_format = quaternion_format
        self.r_weight = Parameter(torch.Tensor(self.in_features, self.out_features))
        self.i_weight = Parameter(torch.Tensor(self.in_features, self.out_features))
        self.j_weight = Parameter(torch.Tensor(self.in_features, self.out_features))
        self.k_weight = Parameter(torch.Tensor(self.in_features, self.out_features))
        self.scale = scale

        if self.scale:
            self.scale_param = Parameter(
                torch.Tensor(self.in_features, self.out_features)
            )
        else:
            self.scale_param = None

        if self.rotation:
            self.zero_kernel = Parameter(
                torch.zeros(self.r_weight.shape), requires_grad=False
            )

        if bias:
            self.bias = Parameter(torch.Tensor(self.out_features * 4))
        else:
            self.register_parameter("bias", None)
        self.init_criterion = init_criterion
        self.weight_init = weight_init
        self.seed = seed if seed is not None else np.random.randint(0, 1234)
        self.rng = RandomState(self.seed)
        self.reset_parameters()

    def reset_parameters(self):
        winit = {
            "quaternion": quaternion_init,
            "unitary": unitary_init,
            "random": random_init,
        }[self.weight_init]
        if self.scale_param is not None:
            torch.nn.init.xavier_uniform_(self.scale_param.data)
        if self.bias is not None:
            self.bias.data.fill_(0)
        affect_init(
            self.r_weight,
            self.i_weight,
            self.j_weight,
            self.k_weight,
            winit,
            self.rng,
            self.init_criterion,
        )

    def forward(self, input):
        # See the autograd section for explanation of what happens here.
        if self.rotation:
            return quaternion_linear_rotation(
                input,
                self.zero_kernel,
                self.r_weight,
                self.i_weight,
                self.j_weight,
                self.k_weight,
                self.bias,
                self.quaternion_format,
                self.scale_param,
            )
        else:
            return quaternion_linear(
                input,
                self.r_weight,
                self.i_weight,
                self.j_weight,
                self.k_weight,
                self.bias,
            )

    def __repr__(self):
        return (
            self.__class__.__name__
            + "("
            + "in_features="
            + str(self.in_features)
            + ", out_features="
            + str(self.out_features)
            + ", bias="
            + str(self.bias is not None)
            + ", init_criterion="
            + str(self.init_criterion)
            + ", weight_init="
            + str(self.weight_init)
            + ", rotation="
            + str(self.rotation)
            + ", seed="
            + str(self.seed)
            + ")"
        )


class QuaternionLinear(Module):
    r"""Applies a quaternion linear transformation to the incoming data."""

    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        init_criterion="he",
        weight_init="quaternion",
        seed=None,
    ):

        super(QuaternionLinear, self).__init__()
        self.in_features = in_features // 4
        self.out_features = out_features // 4
        self.r_weight = Parameter(torch.Tensor(self.in_features, self.out_features))
        self.i_weight = Parameter(torch.Tensor(self.in_features, self.out_features))
        self.j_weight = Parameter(torch.Tensor(self.in_features, self.out_features))
        self.k_weight = Parameter(torch.Tensor(self.in_features, self.out_features))

        if bias:
            self.bias = Parameter(torch.Tensor(self.out_features * 4))
        else:
            self.register_parameter("bias", None)

        self.init_criterion = init_criterion
        self.weight_init = weight_init
        self.seed = seed if seed is not None else np.random.randint(0, 1234)
        self.rng = RandomState(self.seed)
        self.reset_parameters()

    def reset_parameters(self):
        winit = {"quaternion": quaternion_init, "unitary": unitary_init}[
            self.weight_init
        ]
        if self.bias is not None:
            self.bias.data.fill_(0)
        affect_init(
            self.r_weight,
            self.i_weight,
            self.j_weight,
            self.k_weight,
            winit,
            self.rng,
            self.init_criterion,
        )

    def forward(self, input):
        # See the autograd section for explanation of what happens here.
        if input.dim() == 3:
            T, N, C = input.size()
            input = input.view(T * N, C)
            output = QuaternionLinearFunction.apply(
                input,
                self.r_weight,
                self.i_weight,
                self.j_weight,
                self.k_weight,
                self.bias,
            )
            output = output.view(T, N, output.size(1))
        elif input.dim() == 2:
            output = QuaternionLinearFunction.apply(
                input,
                self.r_weight,
                self.i_weight,
                self.j_weight,
                self.k_weight,
                self.bias,
            )
        else:
            raise NotImplementedError

        return output

    def __repr__(self):
        return (
            self.__class__.__name__
            + "("
            + "in_features="
            + str(self.in_features)
            + ", out_features="
            + str(self.out_features)
            + ", bias="
            + str(self.bias is not None)
            + ", init_criterion="
            + str(self.init_criterion)
            + ", weight_init="
            + str(self.weight_init)
            + ", seed="
            + str(self.seed)
            + ")"
        )


class QuaternionAverageMerge(Module):
    r"""Averages two quaternion feature maps, assuming they are aligned."""

    def __init__(self, feat=64):
        super(QuaternionAverageMerge, self).__init__()
        self.feat = feat

    def forward(self, x1, x2):
        # x1 is B,C,2H,2W
        # x2 is B,C,2H,2W
        # We assume x1 and x2 are aligned, i.e., x1 covers even rows/cols and x2 covers odd rows/cols.
        # We average the overlapping pixels for smoothness.
        B, C, H, W = x1.shape
        out = torch.zeros((B, C, 2 * H, 2 * W), dtype=x1.dtype, device=x1.device)
        mask = torch.zeros_like(out)

        # Fill even indices with x1
        out[:, :, ::2, ::2] = x1
        # Fill odd indices with x2, which is offset by 1.
        # if x2 exceeds the bounds, it should be ignored
        out[:, :, 1::2, 1::2] = x2[:, :, :H, :W]  # Ensure x2 is sliced to match output size

        # x = out
        return out

    def __repr__(self):
        return self.__class__.__name__ + "(feat=" + str(self.feat) + ")"
    

class Quaternion2Dslerp(Module):
    """Applies a 2D Slerp Upsampling to the incoming data."""

    def __init__(self, n_feats, upscale_factor):
        super(Quaternion2Dslerp, self).__init__()
        self.n_feats = n_feats
        self.upscale_factor = upscale_factor

    def slerp(self, q1, q2, t):
        """
        Spherical linear interpolation between quaternions q1 and q2.
        q1, q2: (..., 4)
        t: broadcastable tensor for interpolation weight
        """
        # Ensure quaternions are normalized, but avoid zero vectors
        q1_norm = torch.norm(q1, p=2, dim=-1, keepdim=True)
        q2_norm = torch.norm(q2, p=2, dim=-1, keepdim=True)
        
        # Prevent division by zero in normalization
        q1_norm = torch.clamp(q1_norm, min=1e-8)
        q2_norm = torch.clamp(q2_norm, min=1e-8)
        
        q1 = q1 / q1_norm
        q2 = q2 / q2_norm
        
        dot = torch.sum(q1 * q2, dim=-1, keepdim=True)
        q2 = torch.where(dot < 0, -q2, q2)
        dot = torch.abs(dot)
        
        # Clamp dot product more conservatively to avoid numerical issues
        dot = torch.clamp(dot, 0.0, 1.0 - 1e-7)
        theta = torch.acos(dot)
        sin_theta = torch.sin(theta)
        
        # Use more conservative threshold for linear interpolation
        use_lerp = sin_theta < 1e-4
        
        # Make sure t has the same number of dimensions as dot and can broadcast properly
        while t.dim() < dot.dim():
            t = t.unsqueeze(-1)
        
        # Prevent division by zero with larger epsilon
        sin_theta_safe = torch.clamp(sin_theta, min=1e-6)
        
        w1 = torch.sin((1 - t) * theta) / sin_theta_safe
        w2 = torch.sin(t * theta) / sin_theta_safe
        w1_lerp = 1 - t
        w2_lerp = t
        
        w1 = torch.where(use_lerp, w1_lerp, w1)
        w2 = torch.where(use_lerp, w2_lerp, w2)
        
        result = w1 * q1 + w2 * q2
        
        # Ensure result is normalized to prevent accumulation of numerical errors
        result_norm = torch.norm(result, p=2, dim=-1, keepdim=True)
        result_norm = torch.clamp(result_norm, min=1e-8)
        result = result / result_norm
        
        return result

    def forward(self, x):
        # x: (B, C, H, W), C % 4 == 0
        B, C, H, W = x.shape
        upscale = self.upscale_factor
        C4 = C // 4
        x_quat = x.view(B, C4, 4, H, W)  # (B, C4, 4, H, W)

        # Prepare grid for upsampling
        new_H, new_W = H * upscale, W * upscale
        device = x.device
        dtype = x.dtype

        # Generate normalized coordinates for the upsampled grid
        grid_y = torch.linspace(0, H - 1, new_H, device=device)
        grid_x = torch.linspace(0, W - 1, new_W, device=device)
        grid_y, grid_x = torch.meshgrid(grid_y, grid_x, indexing='ij')  # (new_H, new_W)

        y0 = torch.floor(grid_y).long().clamp(0, H - 1)
        x0 = torch.floor(grid_x).long().clamp(0, W - 1)
        y1 = (y0 + 1).clamp(0, H - 1)
        x1 = (x0 + 1).clamp(0, W - 1)

        wy = (grid_y - y0.float()).unsqueeze(0).unsqueeze(0).unsqueeze(0)  # (1,1,1,new_H,new_W)
        wx = (grid_x - x0.float()).unsqueeze(0).unsqueeze(0).unsqueeze(0)  # (1,1,1,new_H,new_W)

        # Gather the 4 corner quaternions for each output location
        def gather_quat(y_idx, x_idx):
            # y_idx, x_idx: (new_H, new_W)
            idx = y_idx * W + x_idx  # flatten index
            flat = x_quat.view(B, C4, 4, H * W)
            out = flat[..., idx.view(-1)].view(B, C4, 4, new_H, new_W)
            return out

        q00 = gather_quat(y0, x0)  # top-left
        q01 = gather_quat(y0, x1)  # top-right
        q10 = gather_quat(y1, x0)  # bottom-left
        q11 = gather_quat(y1, x1)  # bottom-right

        # SLERP along y (vertical)
        q0 = self.slerp(q00, q10, wy)  # left edge
        q1 = self.slerp(q01, q11, wy)  # right edge

        # SLERP along x (horizontal)
        q_final = self.slerp(q0, q1, wx)  # (B, C4, 4, new_H, new_W)

        return q_final.view(B, C, new_H, new_W)

    def __repr__(self):
        return self.__class__.__name__ + "(upscale_factor=" + str(self.upscale_factor) + ")"
