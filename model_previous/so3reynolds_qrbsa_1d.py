import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from model.quat_utils.Qops_with_QSN import conv2d, Residual_SA
from einops import rearrange 
# ─── requirements ───────────────────────────────────────────────────────────────
# pip install torch e3nn==0.7.4              # e3nn just for rotation utilities
# ────────────────────────────────────────────────────────────────────────────────
from e3nn import o3
from model.qrbsa_1d import QRBSA_1D
import scipy.spatial.transform
import numpy as np
import sympy as sp


# # -----------------------------------------------------------------------------#
# # 2.  Quaternion algebra utilities                                             #
# # -----------------------------------------------------------------------------#
# def quat_mul(q1, q2):
#     """
#     Hamilton product of two quaternions.
#     q1: (...,4)   q2: (...,4)   → (...,4)
#     """
#     w1,x1,y1,z1 = q1.unbind(-1)
#     w2,x2,y2,z2 = q2.unbind(-1)
#     return torch.stack([
#         w1*w2 - x1*x2 - y1*y2 - z1*z2,
#         w1*x2 + x1*w2 + y1*z2 - z1*y2,
#         w1*y2 - x1*z2 + y1*w2 + z1*x2,
#         w1*z2 + x1*y2 - y1*x2 + z1*w2
#     ], dim=-1)

# def canonical_quat(q):
#     """Ensure scalar part non‑negative so q ≡ −q."""
#     mask = (q[..., 0:1] < 0).float()
#     return q * (1 - 2*mask)

# # -----------------------------------------------------------------------------#
# # 3.  Rotate *input* quaternion field by left multiplication                   #
# # -----------------------------------------------------------------------------#
# def rotate_input_quat(x, qR, quat_idx=(0,1,2,3)):
#     """
#     x: (B,C,H,W)  – first 4 channels are quaternion [s,x,y,z]
#     qR: (4,)      – group element quaternion
#     quat_idx: channels that form the quaternion
#     """
#     q = x[:, quat_idx, ...].permute(0,2,3,1).contiguous()     # (B,H,W,4)
#     q_rot = quat_mul(qR, q)                                   # left‑mult
#     q_rot = q_rot.permute(0,3,1,2)                            # back to (B,4,H,W)
#     x_r = x.clone()
#     x_r[:, quat_idx, ...] = q_rot
#     return x_r
# # --------‑‑ wrapper itself ‑‑---------------------------------------------------

# def get_full_octahedral_group():
#     # Generate all 24 rotation matrices of the octahedral group using scipy
#     group = scipy.spatial.transform.Rotation.create_group('O')
#     print("group,", group)

#     return torch.tensor(group.as_matrix(), dtype=torch.float32)  # (24, 3, 3)

# #ROT_MATS = get_full_octahedral_group()

# # Quaternion Matrix Expression form
# a, b, c, d = sp.symbols('a b c d')
# quat_matrix_expr = sp.Matrix([[a, -b, -c, -d], [b, a, -d, c], [c ,d ,a, -b], [d, -c, b, a]])

# # FCC Symms Matrix
# h, i, = sp.symbols('half inv_srqt_2')
# fcc_symms = sp.Matrix([[1, 0, 0, 0],
# [0, 1, 0, 0],
# [0, 0, 1, 0],
# [0, 0, 0, 1],
# [i, i, 0, 0 ],
# [i, 0, i, 0],
# [i, 0, 0, i],
# [i, -i, 0, 0],
# [i, 0, -i, 0],
# [i, 0, 0, -i],
# [0, i, i, 0],
# [0, i, 0, i],
# [0, 0, i, i],
# [0, i, -i, 0],
# [0, 0, i, -i],
# [0, i, 0, -i],
# [h, h, h, h],
# [h, -h, -h, h],
# [h, -h, h, -h],
# [h, h, -h, -h],
# [h, h, h, -h],
# [h, h, -h, h],
# [h, -h, h, h],
# [h, -h, -h, -h]])

# # Operators will be generated here and stored as a Python list of SymPy Matrix elements
# # SymPy subs() along with numpy numpy.array().astype() can be used to convert these into numpy arrays
# operator_list = []

# for mat_row in np.arange(sp.shape(fcc_symms)[0]):
#     current_quat = fcc_symms.row(mat_row)
#     current_operator = quat_matrix_expr.subs({a: current_quat[0], b: current_quat[1], c: current_quat[2], d: current_quat[3]})
#     operator_list.append(current_operator)

# # Substitute h=1/2 and i=1/np.sqrt(2) before converting to numpy arrays
# h_val = 1/2
# i_val = 1/np.sqrt(2)
# R = [np.array(op.subs({h: h_val, i: i_val})).astype(np.float32) for op in operator_list]

# def make_model(args):
#     return so3reynolds_qrbsa_1d(args, rot_mats=R, vec_idx=(0, 1, 2), canonicalise=True)

# class so3reynolds_qrbsa_1d(nn.Module):
#     def __init__(self, args,
#                  rot_mats=R,
#                  vec_idx=(0, 1, 2),
#                  canonicalise=True):
#         super().__init__()
#         self.backbone = QRBSA_1D(args)
#         # Convert list of numpy arrays to a torch tensor
#         self.R = torch.tensor(np.stack(rot_mats), dtype=torch.float32)
#         self.vec_idx = vec_idx
#         self.canon = canonicalise

#     def forward(self, x):
#         outs = []
#         for R in self.R.to(x.device):
#             x_r = apply_rotation(x, R, self.vec_idx)
#             o   = self.backbone(x_r)
#             if self.canon:
#                 o = canonical_quat(o)            # ensures one‑to‑one SO3 rep
#             outs.append(o)
#         return torch.stack(outs, 0).mean(0)       # same shape as backbone output


# -----------------------------------------------------------------------------#
# 1.  Build the 24 FCC quaternion‑operator matrices                            #
# -----------------------------------------------------------------------------#
def build_fcc_ops():
    # quaternion symbols
    a, b, c, d = sp.symbols('a b c d')
    quat_mat = sp.Matrix([
        [a, -b, -c, -d],
        [b,  a, -d,  c],
        [c,  d,  a, -b],
        [d, -c,  b,  a]
    ])

    # FCC symmetry table (quaternions)
    h, i = sp.symbols('half inv_sqrt_2')
    fcc = sp.Matrix([
        [1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],
        [i, i,0,0],[i,0, i,0],[i,0,0, i],
        [i,-i,0,0],[i,0,-i,0],[i,0,0,-i],
        [0, i, i,0],[0, i,0, i],[0,0, i, i],
        [0, i,-i,0],[0,0, i,-i],[0, i,0,-i],
        [h, h, h, h],[h,-h,-h, h],[h,-h, h,-h],[h, h,-h,-h],
        [h, h, h,-h],[h, h,-h, h],[h,-h, h, h],[h,-h,-h,-h]
    ])
    subs = {h:0.5, i:1/np.sqrt(2)}

    ops = []
    for r in range(fcc.shape[0]):
        q = fcc.row(r)
        M = quat_mat.subs({a:q[0], b:q[1], c:q[2], d:q[3]}).subs(subs)
        ops.append(np.array(M).astype(np.float32))

    return torch.tensor(ops)          # (24,4,4)

ROT_OPS = build_fcc_ops()             # global constant (24,4,4)

# -----------------------------------------------------------------------------#
# 2.  Utilities                                                                #
# -----------------------------------------------------------------------------#
def rotate_input_quat_mat(x, M, quat_idx=(0,1,2,3)):
    """
    Apply 4×4 operator M to every quaternion in x.
    x: (B,C,H,W)  – channels quat_idx are [s,x,y,z]
    M: (4,4)
    """
    q = x[:, quat_idx, ...].permute(0,2,3,1).contiguous()       # (B,H,W,4)
    q_rot = torch.einsum('ij,bhwj->bhwi', M, q)                 # left‑mult
    q_rot = q_rot.permute(0,3,1,2)                              # (B,4,H,W)
    x_r = x.clone()
    x_r[:, quat_idx, ...] = q_rot
    return x_r

# ------------------------------------------------------------------
# Replace the old rotate_output_mat with the universal version below
# ------------------------------------------------------------------
def rotate_output_mat(o, M):
    """
    Apply the 4×4 operator M to a batch of quaternions *whatever* the
    trailing spatial dimensions are.

    • o can be (B,4) or (B,4,H,W) or (B,4,…) of any rank ≥2.
    • Returns a tensor with the same shape as o, rotated by left‑multiplication.
    """
    if o.dim() == 2:                                  # (B,4)
        return torch.matmul(o, M.T)                   # (B,4)

    # ≥3‑D: move quaternion channel to the last axis → einsum → restore
    # Example: (B,4,H,W) → (B,H,W,4)
    perm_to_last = list(range(0, o.dim()))
    perm_to_last.append(perm_to_last.pop(1))          # move channel idx 1 to end
    o_last = o.permute(*perm_to_last)                 # (...,4)

    # Left‑multiply:  out[...,i] = Σ_j M[i,j] * o_last[...,j]
    o_rot = torch.einsum('ij,...j->...i', M, o_last)  # same shape as o_last

    # Restore original channel order
    inv_perm = list(range(0, o.dim()))
    inv_perm.insert(1, inv_perm.pop())                # move last axis back to 1
    return o_rot.permute(*inv_perm)                   # (B,4,H,W,…)


def canonical_quat(q):
    """Make scalar part ≥ 0 so q ≡ −q."""
    mask = (q[...,0:1] < 0).float()
    return q * (1 - 2*mask)

# -----------------------------------------------------------------------------#
# 3.  SO(3) Reynolds wrapper (matrix version)                                  #
# -----------------------------------------------------------------------------#

def make_model(args):
    """
    Returns an SO(3)‑invariant network whose backbone is QRBSA_1D
    and whose symmetry handling uses 4×4 quaternion operators.
    """
    return so3reynolds_qrbsa_1d(args)

class so3reynolds_qrbsa_1d(nn.Module):
    """
    Invariant network:
        f̂(x) = (1/24) Σ_{M∈H} canonical[ M · f(M · x) ]
    """
    def __init__(self, args,
                 rot_ops=ROT_OPS,
                 quat_idx=(0,1,2,3),
                 canonicalise=True):
        super().__init__()
        self.backbone  = QRBSA_1D(args)
        self.ops       = rot_ops       # (24,4,4)
        self.q_idx     = quat_idx
        self.canon     = canonicalise

    def forward(self, x):
        outs = []
        for M in self.ops.to(x.device):
            x_r = rotate_input_quat_mat(x, M, self.q_idx)  # rotate input
            o   = self.backbone(x_r)                       # (B,4)
            o   = rotate_output_mat(o, M)                  # rotate output
            if self.canon:  o = canonical_quat(o)
            outs.append(o)
        return torch.stack(outs,0).mean(0)                 # invariant (B,4)
    

