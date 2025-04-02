from math import pi
import torch
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch.nn.functional as F
from mat_sci_torch_quats.quats import inverse_matrix_generate, matrix_hamilton_prod, outer_prod, fz_reduce, safe_arccos

def quaternion_inverse(q):
    """
    Returns q⁻¹ for a unit quaternion q by conjugation: q* = [q0, -q1, -q2, -q3].
    Assumes q is already unit length.
    """
    q_inv = q.clone()
    q_inv[..., 1:] = -q_inv[..., 1:]
    return q_inv

def quaternion_multiply(a, b):
    """
    Hamilton product of two quaternions a, b, each shape (..., 4).
    Returns shape (..., 4).
    """
    w1, x1, y1, z1 = a.unbind(dim=-1)
    w2, x2, y2, z2 = b.unbind(dim=-1)
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    return torch.stack([w, x, y, z], dim=-1)

def sym_expand_transform(q1, q2, syms):
    """
    1) Invert q1 => q1_inv
    2) Form T = q2 ⊗ q1⁻¹ (the rotation taking q1 -> q2)
    3) For each symmetry g in syms, compute T_g = T ⊗ g
    4) Pick whichever T_g has the smallest rotation angle about the identity
       (i.e., smallest angle to [1,0,0,0]).
    Returns that 'best' quaternion for each q1,q2 pair in the batch.
    
    Shapes:
      q1, q2:  (..., 4)
      syms:    (n_syms, 4)
    """

    # 1) Invert q1
    q1_inv = quaternion_inverse(q1)

    # 2) Form the base transformation T
    T = matrix_hamilton_prod(q2, q1_inv) # shape (..., 4)

    

    # 3) For each symmetry g in syms, compute T_g = T ⊗ g