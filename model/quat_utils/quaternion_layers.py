##########################################################
# pytorch-qnn v1.0
# Titouan Parcollet
# LIA, Université d'Avignon et des Pays du Vaucluse
# ORKIS, Aix-en-provence
# October 2018
##########################################################

import numpy                   as np
from   numpy.random            import RandomState
import torch
from   torch.autograd           import Variable
import torch.nn.functional      as F
import torch.nn                 as nn
from   torch.nn.parameter       import Parameter
from   torch.nn                 import Module
from   .quaternion_ops          import *
import math
import sys

class QuaternionTransposeConv(Module):
    r"""Applies a Quaternion Transposed Convolution (or Deconvolution) to the incoming data.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 dilatation=1, padding=0, output_padding=0, groups=1, bias=True, init_criterion='he',
                 weight_init='quaternion', seed=None, operation='convolution2d', rotation=False,
                 quaternion_format=False):

        super(QuaternionTransposeConv, self).__init__()

        self.in_channels       = in_channels  // 4
        self.out_channels      = out_channels // 4
        self.stride            = stride
        self.padding           = padding
        self.output_padding    = output_padding
        self.groups            = groups
        self.dilatation        = dilatation
        self.init_criterion    = init_criterion
        self.weight_init       = weight_init
        self.seed              = seed if seed is not None else np.random.randint(0,1234)
        self.rng               = RandomState(self.seed)
        self.operation         = operation
        self.rotation          = rotation
        self.quaternion_format = quaternion_format
        self.winit             = {'quaternion': quaternion_init,
                                  'unitary'   : unitary_init,
                                  'random'    : random_init}[self.weight_init]


        (self.kernel_size, self.w_shape) = get_kernel_and_weight_shape( self.operation,
            self.out_channels, self.in_channels, kernel_size )

        self.r_weight  = Parameter(torch.Tensor(*self.w_shape))
        self.i_weight  = Parameter(torch.Tensor(*self.w_shape))
        self.j_weight  = Parameter(torch.Tensor(*self.w_shape))
        self.k_weight  = Parameter(torch.Tensor(*self.w_shape))


        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        affect_init_conv(self.r_weight, self.i_weight, self.j_weight, self.k_weight,
                    self.kernel_size, self.winit, self.rng, self.init_criterion)
        if self.bias is not None:
           self.bias.data.zero_()

    def forward(self, input):

        if self.rotation:
            return quaternion_tranpose_conv_rotation(input, self.r_weight, self.i_weight,
                self.j_weight, self.k_weight, self.bias, self.stride, self.padding,
                self.output_padding, self.groups, self.dilatation, self.quaternion_format)
        else:
            return quaternion_transpose_conv(input, self.r_weight, self.i_weight, self.j_weight,
                self.k_weight, self.bias, self.stride, self.padding, self.output_padding,
                self.groups, self.dilatation)


    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'in_channels='      + str(self.in_channels) \
            + ', out_channels='   + str(self.out_channels) \
            + ', bias='           + str(self.bias is not None) \
            + ', kernel_size='    + str(self.kernel_size) \
            + ', stride='         + str(self.stride) \
            + ', padding='        + str(self.padding) \
            + ', dilatation='       + str(self.dilatation) \
            + ', init_criterion=' + str(self.init_criterion) \
            + ', weight_init='    + str(self.weight_init) \
            + ', seed='           + str(self.seed) \
            + ', operation='      + str(self.operation) + ')'

class QuaternionConv(Module):
    r"""Applies a Quaternion Convolution to the incoming data.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 dilatation=1, padding=0, groups=1, bias=True, init_criterion='glorot',
                 weight_init='quaternion', seed=None, operation='convolution2d', rotation=False, quaternion_format=True, scale=False):

        super(QuaternionConv, self).__init__()

        self.in_channels       = in_channels  // 4
        self.out_channels      = out_channels // 4
        self.stride            = stride
        self.padding           = padding
        self.groups            = groups
        self.dilatation        = dilatation
        self.init_criterion    = init_criterion
        self.weight_init       = weight_init
        self.seed              = seed if seed is not None else np.random.randint(0,1234)
        self.rng               = RandomState(self.seed)
        self.operation         = operation
        self.rotation          = rotation
        self.quaternion_format = quaternion_format
        self.winit             =    {'quaternion': quaternion_init,
                                     'unitary'   : unitary_init,
                                     'random'    : random_init}[self.weight_init]
        self.scale             = scale


        (self.kernel_size, self.w_shape) = get_kernel_and_weight_shape( self.operation,
            self.in_channels, self.out_channels, kernel_size )

        self.r_weight  = Parameter(torch.Tensor(*self.w_shape))
        self.i_weight  = Parameter(torch.Tensor(*self.w_shape))
        self.j_weight  = Parameter(torch.Tensor(*self.w_shape))
        self.k_weight  = Parameter(torch.Tensor(*self.w_shape))

        if self.scale:
            self.scale_param  = Parameter(torch.Tensor(self.r_weight.shape))
        else:
            self.scale_param  = None

        if self.rotation:
            self.zero_kernel = Parameter(torch.zeros(self.r_weight.shape), requires_grad=False)
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        affect_init_conv(self.r_weight, self.i_weight, self.j_weight, self.k_weight,
                    self.kernel_size, self.winit, self.rng, self.init_criterion)
        if self.scale_param is not None:
            torch.nn.init.xavier_uniform_(self.scale_param.data)
        if self.bias is not None:
           self.bias.data.zero_()

    def forward(self, input):

        #import pdb; pdb.set_trace()
        if self.rotation:
            return quaternion_conv_rotation(input, self.zero_kernel, self.r_weight, self.i_weight, self.j_weight,
                self.k_weight, self.bias, self.stride, self.padding, self.groups, self.dilatation,
                self.quaternion_format, self.scale_param)
        else:
            return quaternion_conv(input, self.r_weight, self.i_weight, self.j_weight,
                self.k_weight, self.bias, self.stride, self.padding, self.groups, self.dilatation)


    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'in_channels='      + str(self.in_channels) \
            + ', out_channels='   + str(self.out_channels) \
            + ', bias='           + str(self.bias is not None) \
            + ', kernel_size='    + str(self.kernel_size) \
            + ', stride='         + str(self.stride) \
            + ', padding='        + str(self.padding) \
            + ', init_criterion=' + str(self.init_criterion) \
            + ', weight_init='    + str(self.weight_init) \
            + ', seed='           + str(self.seed) \
            + ', rotation='       + str(self.rotation) \
            + ', q_format='       + str(self.quaternion_format) \
            + ', operation='      + str(self.operation) + ')'


def quaternion_multiply(q, r): 
    
    """ Multiply two quaternions. Both q and r are tensors with the last dimension = 4, representing [r, i, j, k]. """ 
    # move dim 2 to the end
    q = q.permute(*range(q.dim()-2), -1, -2) if q.dim() > 2 else q
    qr = q[..., 0] 
    qi = q[..., 1] 
    qj = q[..., 2] 
    qk = q[..., 3]

    rr = r[..., 0]
    ri = r[..., 1]
    rj = r[..., 2]
    rk = r[..., 3]

    out_r = qr*rr - qi*ri - qj*rj - qk*rk
    out_i = qr*ri + qi*rr + qj*rk - qk*rj
    out_j = qr*rj - qi*rk + qj*rr + qk*ri
    out_k = qr*rk + qi*rj - qj*ri + qk*rr
    return torch.stack([out_r, out_i, out_j, out_k], dim=-1)


class ChannelRotation(nn.Module): 
    
    """ Applies a channel-dependent quaternion rotation.
    In this module the rotation axis (a unit vector u) and the rotation
    increment (theta) are learned and common across all channels. For each channel
    index i, the rotation quaternion is computed as:
    In this module the rotation axis (a unit vector u) and the rotation
    increment (theta) are learned and common across all channels. For each channel
    index i, the rotation quaternion is computed as:

        qᵢ = [ cos((i·θ)/2),
            uₓ * sin((i·θ)/2),
            u_y * sin((i·θ)/2),
            u_z * sin((i·θ)/2) ]
            
    so that for 4 channels the rotations are 0, θ, 2θ, and 3θ.
    """

    def __init__(self, num_channels):
        super(ChannelRotation, self).__init__()
        self.num_channels = num_channels
        # Learnable scalar theta (initial value π/8)
        self.theta = Parameter(torch.tensor(np.pi/8, dtype=torch.float32))
        # Learnable axis vector u (3D); initialize to [0,0,1] for example.
        self.u = Parameter(torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32))
        
    def forward(self, x):
        # x shape: [B, num_channels, 4, *spatial_dims]
        B = x.size(0)
        spatial_dims = x.size()[3:]
        device = x.device
        dtype = x.dtype
        
        # Normalize u to ensure unit vector
        u_normalized = self.u / (self.u.norm(p=2) + 1e-8)
        # Create channel indices: 0,1,...,num_channels-1 (as float)
        indices = torch.arange(self.num_channels, device=device, dtype=dtype)  # shape: [num_channels]
        # Compute the rotation angles: each channel gets an angle = (i * theta)
        angles = indices * self.theta  # [0, theta, 2theta, ...]
        # Compute cosine and sine (note that the rotation quaternion uses half the angle)
        cos_vals = torch.cos(angles / 2)
        sin_vals = torch.sin(angles / 2)
        # Build the rotation quaternion for each channel:
        # q = [cos(angle/2), uₓ*sin(angle/2), u_y*sin(angle/2), u_z*sin(angle/2)]
        rot_q = torch.stack([cos_vals,
                            u_normalized[0] * sin_vals,
                            u_normalized[1] * sin_vals,
                            u_normalized[2] * sin_vals], dim=1)  # shape: [num_channels, 4]
        # Reshape so that rot_q can broadcast: shape [1, num_channels, 4, 1, 1, ...]
        view_shape = (1, self.num_channels, 4) + (1,)*len(spatial_dims)
        rot_q = rot_q.view(view_shape)
        # Multiply each channel's quaternion output by its rotation quaternion.
        return quaternion_multiply(rot_q, x)


class QuaternionConv_w_rotation(nn.Module): 

    """Applies a Quaternion Convolution to the incoming data.
    This version optionally includes channel-dependent rotation. If 
    channel_dependence is True, after performing the quaternion convolution,
    the module reshapes the output into quaternion channels, and applies a
    ChannelRotation module that learns a common rotation axis u and rotation
    increment theta. The output of channel i is rotated by angle i·theta.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride,
                dilatation=1, padding=0, groups=1, bias=True, init_criterion='glorot',
                weight_init='quaternion', seed=None, operation='convolution2d', rotation=False,
                quaternion_format=True, scale=False,
                channel_dependence=True):
        
        super(QuaternionConv_w_rotation, self).__init__()
        self.in_channels       = in_channels  // 4
        self.out_channels      = out_channels // 4
        self.stride            = stride
        self.padding           = padding
        self.groups            = groups
        self.dilatation        = dilatation
        self.init_criterion    = init_criterion
        self.weight_init       = weight_init
        self.seed              = seed if seed is not None else np.random.randint(0,1234)
        self.rng               = RandomState(self.seed)
        self.operation         = operation
        self.rotation          = rotation  # Determines whether to use quaternion_conv_rotation
        self.quaternion_format = quaternion_format
        self.winit             = {'quaternion': quaternion_init,
                                'unitary'   : unitary_init,
                                'random'    : random_init}[self.weight_init]
        self.scale             = scale

        # New: channel-dependent rotation configuration.
        self.channel_dependence = channel_dependence
        if self.channel_dependence:
            self.channel_rotator = ChannelRotation(self.out_channels)
        else:
            self.channel_rotator = None

        (self.kernel_size, self.w_shape) = get_kernel_and_weight_shape(self.operation,
                                                                        self.in_channels, self.out_channels, kernel_size)

        self.r_weight  = Parameter(torch.Tensor(*self.w_shape))
        self.i_weight  = Parameter(torch.Tensor(*self.w_shape))
        self.j_weight  = Parameter(torch.Tensor(*self.w_shape))
        self.k_weight  = Parameter(torch.Tensor(*self.w_shape))

        if self.scale:
            self.scale_param  = Parameter(torch.Tensor(self.r_weight.shape))
        else:
            self.scale_param  = None

        if self.rotation:
            self.zero_kernel = Parameter(torch.zeros(self.r_weight.shape), requires_grad=False)
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        affect_init_conv(self.r_weight, self.i_weight, self.j_weight, self.k_weight,
                        self.kernel_size, self.winit, self.rng, self.init_criterion)
        if self.scale_param is not None:
            torch.nn.init.xavier_uniform_(self.scale_param.data)
        if self.bias is not None:
            self.bias.data.zero_()

    def forward(self, input):
        # Compute the primary quaternion convolution output.
        if self.rotation:
            Y = quaternion_conv_rotation(input, self.zero_kernel, self.r_weight, self.i_weight, self.j_weight,
                                        self.k_weight, self.bias, self.stride, self.padding, self.groups,
                                        self.dilatation, self.quaternion_format, self.scale_param)
        else:
            Y = quaternion_conv(input, self.r_weight, self.i_weight, self.j_weight,
                                self.k_weight, self.bias, self.stride, self.padding, self.groups, self.dilatation)
        # Reshape to group every 4 channels into a quaternion: [B, out_channels, 4, ...]
        B = Y.size(0)
        spatial_dims = Y.size()[2:]
        Y = Y.view(B, self.out_channels, 4, *spatial_dims)
        # If channel-dependent rotation is enabled, apply it.
        if self.channel_rotator is not None:
            Y = self.channel_rotator(Y)
        # Reshape back to the canonical output form: [B, 4*out_channels, ...]
        Y = Y.view(B, 4 * self.out_channels, *spatial_dims)
        return Y

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'in_channels='      + str(self.in_channels) \
            + ', out_channels='   + str(self.out_channels) \
            + ', bias='           + str(self.bias is not None) \
            + ', kernel_size='    + str(self.kernel_size) \
            + ', stride='         + str(self.stride) \
            + ', padding='        + str(self.padding) \
            + ', init_criterion=' + str(self.init_criterion) \
            + ', weight_init='    + str(self.weight_init) \
            + ', seed='           + str(self.seed) \
            + ', rotation='       + str(self.rotation) \
            + ', q_format='       + str(self.quaternion_format) \
            + ', operation='      + str(self.operation) \
            + ', channel_dependence=' + str(self.channel_dependence) + ')'



class QuaternionLinearAutograd(Module):
    r"""Applies a quaternion linear transformation to the incoming data. A custom
    Autograd function is call to drastically reduce the VRAM consumption. Nonetheless, computing
    time is also slower compared to QuaternionLinear().
    """

    def __init__(self, in_features, out_features, bias=True,
                 init_criterion='glorot', weight_init='quaternion',
                 seed=None, rotation=False, quaternion_format=True, scale=False):

        super(QuaternionLinearAutograd, self).__init__()
        self.in_features       = in_features//4
        self.out_features      = out_features//4
        self.rotation          = rotation
        self.quaternion_format = quaternion_format
        self.r_weight = Parameter(torch.Tensor(self.in_features, self.out_features))
        self.i_weight = Parameter(torch.Tensor(self.in_features, self.out_features))
        self.j_weight = Parameter(torch.Tensor(self.in_features, self.out_features))
        self.k_weight = Parameter(torch.Tensor(self.in_features, self.out_features))
        self.scale    = scale

        if self.scale:
            self.scale_param  = Parameter(torch.Tensor(self.in_features, self.out_features))
        else:
            self.scale_param  = None

        if self.rotation:
            self.zero_kernel  = Parameter(torch.zeros(self.r_weight.shape), requires_grad=False)

        if bias:
            self.bias = Parameter(torch.Tensor(self.out_features*4))
        else:
            self.register_parameter('bias', None)
        self.init_criterion = init_criterion
        self.weight_init = weight_init
        self.seed = seed if seed is not None else np.random.randint(0,1234)
        self.rng = RandomState(self.seed)
        self.reset_parameters()

    def reset_parameters(self):
        winit = {'quaternion': quaternion_init, 'unitary': unitary_init, 'random': random_init}[self.weight_init]
        if self.scale_param is not None:
            torch.nn.init.xavier_uniform_(self.scale_param.data)
        if self.bias is not None:
            self.bias.data.fill_(0)
        affect_init(self.r_weight, self.i_weight, self.j_weight, self.k_weight, winit,
                    self.rng, self.init_criterion)

    def forward(self, input):
        # See the autograd section for explanation of what happens here.
        if self.rotation:
            return quaternion_linear_rotation(input, self.zero_kernel, self.r_weight, self.i_weight, self.j_weight, self.k_weight, self.bias, self.quaternion_format, self.scale_param)
        else:
            return quaternion_linear(input, self.r_weight, self.i_weight, self.j_weight, self.k_weight, self.bias)

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'in_features=' + str(self.in_features) \
            + ', out_features=' + str(self.out_features) \
            + ', bias=' + str(self.bias is not None) \
            + ', init_criterion=' + str(self.init_criterion) \
            + ', weight_init=' + str(self.weight_init) \
            + ', rotation='       + str(self.rotation) \
            + ', seed=' + str(self.seed) + ')'

class QuaternionLinear(Module):
    r"""Applies a quaternion linear transformation to the incoming data.
    """

    def __init__(self, in_features, out_features, bias=True,
                 init_criterion='he', weight_init='quaternion',
                 seed=None):

        super(QuaternionLinear, self).__init__()
        self.in_features  = in_features//4
        self.out_features = out_features//4
        self.r_weight     = Parameter(torch.Tensor(self.in_features, self.out_features))
        self.i_weight     = Parameter(torch.Tensor(self.in_features, self.out_features))
        self.j_weight     = Parameter(torch.Tensor(self.in_features, self.out_features))
        self.k_weight     = Parameter(torch.Tensor(self.in_features, self.out_features))

        if bias:
            self.bias     = Parameter(torch.Tensor(self.out_features*4))
        else:
            self.register_parameter('bias', None)

        self.init_criterion = init_criterion
        self.weight_init    = weight_init
        self.seed           = seed if seed is not None else np.random.randint(0,1234)
        self.rng            = RandomState(self.seed)
        self.reset_parameters()

    def reset_parameters(self):
        winit = {'quaternion': quaternion_init,
                 'unitary': unitary_init}[self.weight_init]
        if self.bias is not None:
            self.bias.data.fill_(0)
        affect_init(self.r_weight, self.i_weight, self.j_weight, self.k_weight, winit,
                    self.rng, self.init_criterion)

    def forward(self, input):
        # See the autograd section for explanation of what happens here.
        if input.dim() == 3:
            T, N, C = input.size()
            input  = input.view(T * N, C)
            output = QuaternionLinearFunction.apply(input, self.r_weight, self.i_weight, self.j_weight, self.k_weight, self.bias)
            output = output.view(T, N, output.size(1))
        elif input.dim() == 2:
            output = QuaternionLinearFunction.apply(input, self.r_weight, self.i_weight, self.j_weight, self.k_weight, self.bias)
        else:
            raise NotImplementedError

        return output

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'in_features=' + str(self.in_features) \
            + ', out_features=' + str(self.out_features) \
            + ', bias=' + str(self.bias is not None) \
            + ', init_criterion=' + str(self.init_criterion) \
            + ', weight_init=' + str(self.weight_init) \
            + ', seed=' + str(self.seed) + ')'
