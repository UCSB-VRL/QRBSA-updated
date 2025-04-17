from math import pi
import torch
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch.nn.functional as F
from mat_sci_torch_quats.symmetries import fcc_syms, hcp_syms

# Defines mapping from quat vector to matrix. Though there are many
# possible matrix representations, this one is selected since the
# first row, X[...,0], is the vector form.
# https://en.wikipedia.org/wiki/Quaternion#Matrix_representations
#import pdb; pdb.set_trace()
q1 = np.diag([1,1,1,1])
qj = np.roll(np.diag([-1,1,1,-1]),-2,axis=1)
qk = np.diag([-1,-1,1,1])[:,::-1]
qi = np.matmul(qj,qk)
Q_arr = torch.Tensor([q1,qi,qj,qk])
Q_arr_flat = Q_arr.reshape((4,16))

num_classes = 48  



def one_hot_encode(symm_vectors, num_classes):
    """
    One-hot encode the categorical symm_vectors.
    
    Args:
        symm_vectors (Tensor): Tensor of shape (B, C, H, W), with C being the categorical class.
        num_classes (int): Number of possible categories (classes) for symmetries.
        
    Returns:
        Tensor: One-hot encoded tensor of shape (B, num_classes, H, W).
    """
    return F.one_hot(symm_vectors.long(), num_classes=num_classes).float()


def compute_hamming_distance(patch1, patch2):
    """
    Compute the Hamming distance between two one-hot encoded patches.
    
    Args:
        patch1 (Tensor): A one-hot encoded patch of shape (num_classes, H', W').
        patch2 (Tensor): Another one-hot encoded patch of the same shape.
        
    Returns:
        Tensor: Hamming distance for each patch.
    """
    return (patch1 != patch2).sum(dim=1)  # Sum across the class dimension to compute the distance


def kernel_symmetry_local_loss(symm_vectors, kernel_size=3, num_classes=48, threshold=0.1):
    """
    Apply a kernel-based loss that penalizes variations in symmetry within each local patch.
    
    Args:
        symm_vectors (Tensor): One-hot encoded tensor of shape (B, H, W, C).
        kernel_size (int): Size of the kernel (e.g., 3 for 3x3 patches).
        num_classes (int): Number of possible categories for symmetries.
        threshold (float): Threshold for penalizing high Hamming distance (more variation).
        
    Returns:
        Tensor: Local kernel loss that penalizes variations within each patch.
    """
    
    # Convert symm_vectors to (B, C, H, W) for F.unfold
    symm_vectors = symm_vectors.permute(0, 3, 1, 2)  # Convert from [B, H, W, C] to [B, C, H, W]

    # Use F.unfold to extract patches (B, C * K * K, H', W')
    unfolded = F.unfold(symm_vectors, kernel_size=kernel_size)

    # Now we are comparing across the K*K (kernel size), so let's compute the Hamming loss within each patch
    # Reshape to (B, num_classes, K*K, H', W') - the last two dimensions will represent the flattened patch
    unfolded_reshaped = unfolded.view(
        unfolded.shape[0],  # Batch size (B)
        num_classes,  # Number of classes (C)
        kernel_size*kernel_size,  # Flatten K*K (9 or whatever kernel size you're using)
        symm_vectors.shape[2] - kernel_size + 1,  # Output height after convolution
        symm_vectors.shape[3] - kernel_size + 1   # Output width after convolution
    )

    # Compute the Hamming distance for the one-hot encoded vectors within the patch.
    # The Hamming distance is computed across the K*K dimension.
    # We use log-softmax to compute the log-likelihood for each pixel
    cross_entropy_loss = F.cross_entropy(
        unfolded_reshaped,              # This represents the predicted "probabilities"
        torch.argmax(unfolded_reshaped, dim=1),  # We are using the argmax to simulate the target class
        reduction="none"  # Compute element-wise loss
    )
    
    # To get a total loss for each patch (across all pixels in the patch), you can sum or average the losses
    patch_loss = cross_entropy_loss.mean(dim=1)  # Sum over K*K (last two dimensions)
    
    return patch_loss


def kernel_misorientation_deviation(qSR, qHR, angles, selected_symmetries, syms=None):
        
        # two losses
        # 1. higher deviation
        # 2. choosing wrong symmetry diffferent from average 

        # take HR where the deviation is negligible in the reduced symmetry
        # SR should be able to reproduce the same - [penalize if not] based on 1, 2

        # Input shape of qSR and qHR is (batch_size, ch, 64, 64)
        #import pdb; pdb.set_trace()

        # run a kernel on qHR of size 3x3 on dim [2, 3] to see where there is no transition of grain boundary
        # check it by comparing the values of the kernel with the central value. If the values are the same, then it is not a grain boundary.
        # if the max-min difference is less than 5 degrees, then it is not a grain boundary.
        # if the values are different are less than 5degrees, then it is not a grain boundary. keep this else continue.

        """
        Compute misorientation deviation in non-grain-boundary regions using quaternion and angle maps.
        
        Args:
                qSR: Super-resolved quaternions, shape (B, 4, H, W)
                qHR: Ground-truth high-res quaternions, shape (B, 4, H, W)
                angles: Misorientation angles (in radians), shape (B, H, W)
        
        Returns:
                misor_diff: Misorientation deviation for non-boundary regions, shape (B, H-2, W-2)
        """
        
        qHRperm = qHR.permute(0, 3, 1, 2).contiguous()

        K = 3  # Kernel size
        threshold = 0.0872665  # ~5 degrees in radians
        B, C, H, W = qHRperm.shape  

        # Extract 3x3 patches using unfold
        
        q_patches = F.unfold(qHRperm, kernel_size=K).view(B, C, K * K, H - (K-1), W - (K-1))
        angles_patches = F.unfold(angles.unsqueeze(1), kernel_size=K).view(B, K * K, H - (K-1), W - (K-1))

        # Get orientation angles from scalar part of quaternions
        q0_patches = q_patches[:, 0]  # Shape: (B, 9, H-2, W-2)
        orientation_angles = 2 * torch.arccos(q0_patches.clamp(-1 + 1e-7, 1 - 1e-7))

        # HERE THE ANGLES ARE CALCULATED WRT [1000] -- BASIS OF IPF MAP COLOR GENERATION. 
        # CHOOSE THE SYMMETRY WITH RIGHT 

        # Get the angles of the patches with right symmetry
        if syms is not None:
                syms = syms.to(qHR.device)
                qpatches_inv = inverse_matrix_generate(q_patches) # only uses q1 to obtain tensor shape.
                # unit_quat = torch.Tensor([1,0,0,0]).to(qHR.device)
                # 1) Create a [1, 4, 1, 1, 1] tensor so that the second dimension is 4
                unit_quat = torch.zeros((1, 4, 1, 1, 1), device=qHR.device)
                unit_quat[..., 0] = 1.0  # make it the identity quaternion: [1,0,0,0]

                # 2) Expand to [B, C=4, K*K, H-(K-1), W-(K-1)]
                #    => e.g. [4, 4, 9, 62, 62]
                unit_quat = unit_quat.expand(B, 4, K*K, H - (K - 1), W - (K - 1))

                T1 = matrix_hamilton_prod(qpatches_inv.permute(0,2,3,4,1).contiguous(), unit_quat.permute(0, 2,3,4,1).contiguous())

                T_syms = outer_prod(T1, syms)
                T_syms = T_syms.view(-1, syms.shape[0], 4)
                
                # import pdb; pdb.set_trace()
                orientation_angles = 2*torch.arccos(T_syms[...,0])
                # get the minimum angle
                min_ind = orientation_angles.min(-1)[1] # still differentiable --> gradient flows through only for min.
                min_ind_flat = min_ind.view(-1)
                orientation_angles = orientation_angles[torch.arange(len(orientation_angles)), min_ind_flat]
                orientation_angles = orientation_angles.view(B, K*K, H - (K-1), W - (K-1))      

        # Determine non-grain-boundary patches
        min_angle = orientation_angles.min(dim=1).values
        max_angle = orientation_angles.max(dim=1).values
        mask = (max_angle - min_angle) < threshold  # Shape: (B, H-2, W-2)

        # Misorientation spread
        min_misor = angles_patches.min(dim=1).values
        max_misor = angles_patches.max(dim=1).values
        misor_diff = max_misor - min_misor  # (B, H-2, W-2)
        
        # Zero out regions not satisfying the mask
        misor_diff = misor_diff * mask

        # NOW get variance in kernel symmetry zone selection
        # import pdb; pdb.set_trace()
        selected_symmetries= one_hot_encode(selected_symmetries, num_classes)
        selected_symm_kernel_var = kernel_symmetry_local_loss(selected_symmetries, num_classes=num_classes, kernel_size=K)
        selected_symm_kernel_var= selected_symm_kernel_var * mask

        kernel_loss = selected_symm_kernel_var + misor_diff


        return kernel_loss


# Checks if 2 arrays can be broadcast together
def _broadcastable(s1,s2):
        if len(s1) != len(s2): return False
        else: return all((i==j) or (i==1) or (j==1) for i,j in zip(s1,s2))

# Converts an array of quats as vectors to matrices. Generally
# used to facilitate quat multiplication.
def vec2mat(X):
        #import pdb; pdb.set_trace()
        assert X.shape[-1] == 4, 'Last dimension must be of size 4'
        new_shape = X.shape[:-1] + (4,4)
        dtype = X.dtype
        Q = Q_arr_flat.type(X.dtype).to(X.device)
        #print('Q', Q.dtype)
        return torch.matmul(X,Q).reshape(new_shape)


# Performs element-wise multiplication, like the standard multiply in
# numpy. Equivalent to q1 * q2.
def hadamard_prod(q1,q2):
        assert _broadcastable(q1.shape,q2.shape), 'Inputs of shapes ' \
                        f'{q1.shape}, {q2.shape} could not be broadcast together'
        X1 = vec2mat(q1)
        X_out = (X1 * q2[...,None,:]).sum(-1)
        return X_out


# Performs outer product on ndarrays of quats
# Ex if X1.shape = (s1,s2,4) and X2.shape = (s3,s4,s5,4),
# output will be of size (s1,s2,s3,s4,s5,4)
def outer_prod(q1,q2):
        #import pdb; pdb.set_trace()
        X1 = vec2mat(q1)
        X2 = torch.movedim(q2,-1,0)
        X1_flat = X1.reshape((-1,4))
        X2_flat = X2.reshape((4,-1))
        X_out = torch.matmul(X1_flat,X2_flat)
        X_out = X_out.reshape(q1.shape + q2.shape[:-1])
        X_out = torch.movedim(X_out,len(q1.shape)-1,-1)
        return X_out


# Utilities to create random vectors on the L2 sphere. First produces
# random samples from a rotationally invariantt distibution (i.e. Gaussian)
# and then normalizes onto the unit sphere

# Produces random array of the same size as shape.
def rand_arr(shape,dtype=torch.FloatTensor):
        if not isinstance(shape,tuple): shape = (shape,)
        X = torch.randn(shape).type(dtype)
        X /= torch.norm(X,dim=-1,keepdim=True)
        return X

# Produces array of 3D points on the unit sphere.
def rand_points(shape,dtype=torch.FloatTensor):
        if not isinstance(shape,tuple): shape = (shape,)
        return rand_arr(shape + (3,), dtype)

# Produces random unit quaternions.
def rand_quats(shape,dtype=torch.FloatTensor):
        if not isinstance(shape,tuple): shape = (shape,)
        return rand_arr(shape+(4,), dtype)


# arccos, expanded from range [-1,1] to all real numbers
# values outside of [-1,1] and replaced with a line of slope pi/2, such that
# the function is continuous
def safe_arccos(x):
    mask = (torch.abs(x) < 1).float()
    x_clip = torch.clamp(x,min=-1,max=1)
    output_arccos = torch.arccos(x_clip)
    output_linear = (1 - x)*pi/2
    output = mask*output_arccos + (1-mask)*output_linear
    return output


# arccos, expanded from range [-1,1] to all real numbers
# values outside of [-1,1] and replaced with a line of slope pi/2, such that
# the function is continuous
def safe_arccos(x):
    mask = (torch.abs(x) < 1).float()
    x_clip = torch.clamp(x,min=-1,max=1)
    output_arccos = torch.arccos(x_clip)
    output_linear = (1 - x)*pi/2
    output = mask*output_arccos + (1-mask)*output_linear
    return output

# Generate the minimum angle transformation with PyTorch, to enable automatic differentiation
def transformation_matrix_tensor(q1, q2, syms):

        #syms_neg = -1*syms
        #syms = torch.cat((syms, syms_neg))

        # VERIFIED THAT SYMS NEG GIVES THE SAME VALUES AGAIN.
        B,H,W,C = q1.shape
        syms = syms.to(q1.device)

        q1_inv = inverse_matrix_generate(q1) # only uses q1 to obtain tensor shape.
        #q2_inv= inverse_matrix_generate(q2) # only uses q2 to obtain tensor shape.

        # VERIFIED THAT REAL PART OF T1 AND T2 ARE THE SAME. SAME ROTATION ANGLE.

        T1 = matrix_hamilton_prod(q1_inv, q2)
        #T2 = matrix_hamilton_prod(q2_inv, q1)

        T1_syms = outer_prod(T1, syms)
        T1_syms = T1_syms.view(-1, syms.shape[0], 4)

        #T2_syms = outer_prod(T2, syms)
        #T2_syms = T2_syms.view(-1, syms.shape[0], 4)

        ## Is it possible 
        # import pdb; pdb.set_trace()
        #T_syms = torch.cat((T1_syms, T2_syms), 1)

        T_syms=T1_syms
        theta = 2*torch.arccos(T_syms[...,0])
        min_ind = theta.min(-1)[1] # still differentiable --> gradient flows through only for min.
        min_ind_flat = min_ind.view(-1)

        try:
                # import pdb; pdb.set_trace()
                T_min = T_syms[torch.arange(len(T_syms)), min_ind_flat]

        except RuntimeError as e:
                print('broadcasting issue \n')
                #import pdb; pdb.set_trace()

        T_min = T_min.reshape(q1.shape)
        min_ind=min_ind.reshape(B,H,W)

        # q_loss_inv = matrix_hamilton_prod(q1_inv, T_min) ## Perhaps the error is here, can't backpropagate current inverse function applied to q_nn.
        # q_loss = q_loss_inv * inv
        return T_min, min_ind

# Generate an "inverse-creating" tensor (will generate an inverse when multiplied with quaternion orientation tensor) required for the size of input matrix
def inverse_matrix_generate(q):
    """
    Returns the inverse q^-1 = conjugate(q) / (||q||^2), 
    for each quaternion in q. 
    q: shape (..., 4) => last dimension is [w, x, y, z].
    
    We assume q might not be normalized. If q is guaranteed unit, 
    then q^-1 = conjugate(q), ignoring the norm^2 factor.
    """
    # norm^2 for each quaternion
    norm_sq = (q * q).sum(dim=-1, keepdim=True)  # shape (...,1)
    # or:  norm_sq = q.pow(2).sum(...)

    # clone q, flip the sign of x,y,z
    q_inv = q.clone()
    q_inv[..., 1:] *= -1  # conjugate

    # divide by norm^2
    q_inv = q_inv / norm_sq.clamp_min(1e-15)

    return q_inv


## ! issue was likely here, make sure this is performed as differentiable matrix operation
def inverse(q):
        # import pdb; pdb.set_trace()
        magnitudes = torch.norm(q,2,-1)
        q_inv = q.clone()
        q_inv[...,1:4] = -1 * q[...,1:4]
        q_inv = 1/magnitudes.unsqueeze(-1) * q_inv

        return q_inv


# Matrix Hamilton product
def matrix_hamilton_prod(q1,q2):
        assert _broadcastable(q1.shape,q2.shape), 'Inputs of shapes ' \
                        f'{q1.shape}, {q2.shape} could not be broadcast together'

        # q2 = q2.to('cuda:0')
        X1 = vec2mat(q1)
        X_out = (X1 * q2[...,None,:]).sum(-1)
        return X_out

# Calculates validation loss, using the minimum angle transformation, but without tracking gradients.
def validation_min_angle_transformation(q1, q2, syms):

        device = q1.device
        T = matrix_hamilton_prod(q1, inverse(q2.to(device)))
        T_syms = outer_prod(T, syms)
        T_syms = T_syms.view(-1, syms.shape[0], 4)

        # T2 = matrix_hamilton_prod(q2, inverse(q1.to(device)))
        # T2_syms = outer_prod(T2, syms)
        # T2_syms = T2_syms.view(-1, syms.shape[0], 4)

        #T_syms = torch.cat((T1_syms, T2_syms), 1)

        theta = torch.arccos(T_syms[...,0])
        min_ind = theta.min(-1)[1]

        # theta_min = theta[torch.arange(len(theta)), min_ind]
        # import pdb; pdb.set_trace()
        T_min = T_syms[torch.arange(len(T_syms)), min_ind]
        T_min = T_min.reshape(q1.shape)

        theta = 2*safe_arccos(T_min[...,0])
        #theta = 2*torch.arccos(T_min[...,0])
        # zero_broadcast_tensor = torch.Tensor([1,0,0,0])
        # zero_broadcast_tensor = zero_broadcast_tensor.reshape(1,1,1,4).to(torch.device('cuda:0'))

        # euclid_dist = torch.linalg.norm(T_min - zero_broadcast_tensor, 2, dim=-1)
        # # import pdb; pdb.set_trace() ## WHY DID I PLACE A 0 INDEX?
        # dist = 4*torch.arcsin(euclid_dist / 2)
        return theta

def validation_rot_dist_approx_MAT_symmetry(q1, q2, syms):
        device = torch.device('cuda:0')

        q1 = q1.to(device)
        q2 = q2.to(device)
        T1 = matrix_hamilton_prod(q1, inverse(q2.to(device)))
        T1_syms = outer_prod(T1, syms)
        T1_syms = T1_syms.view(-1, syms.shape[0], 4)

        theta = 2*safe_arccos(T1_syms[...,0])
        min_ind = theta.min(-1)[1]

        # theta_min = theta[torch.arange(len(theta)), min_ind]
        # import pdb; pdb.set_trace()
        T_min = T1_syms[torch.arange(len(T1_syms)), min_ind]
        T_min = T_min.reshape(q1.shape)

        theta = 2*safe_arccos(T_min[...,0])

        return theta

def quat_dist(q1,q2=None):
        """
        Computes distance between two quats. If q1 and q2 are on the unit sphere,
        this will return the arc length along the sphere. For points within the
        sphere, it reduces to a function of MSE.
        """
        import pdb; pdb.set_trace()
        if q2 is None: mse = (q1[...,0]-1)**2 + (q1[...,1:]**2).sum(-1)
        else: mse = ((q1-q2)**2).sum(-1)
        
        corr = 1 - (1/2)*mse
        # my fz version
        #corr = 1-2*(1- q1[...,0]**2)
    
        #corr_clamp = torch.clamp(corr,min=-1,max=1)
        #arccos = torch.arccos(corr_clamp)
        return safe_arccos(corr)
        #return arccos 
        

def rot_dist(q1,q2=None):
        """ Get dist between two rotations, with q <-> -q symmetry """
        #import pdb; pdb.set_trace()
        q1_w_neg = torch.stack((q1,-q1),dim=-2)
        if q2 is not None: q2 = q2[...,None,:]
        dists = quat_dist(q1_w_neg,q2)
        dist_min = dists.min(-1)[0]
        return dist_min

        
def fz_reduce(q,syms):
        #import pdb; pdb.set_trace()
        shape = q.shape
        q = q.reshape((-1,4))
        syms = syms.cuda()
        q_w_syms = outer_prod(q, syms)
        real_part = q_w_syms[..., 0].clamp(min=-1.0, max=1.0)
        theta = torch.arccos(real_part)  # shape: (B, N)

        # 2) Find which symmetry yields the minimal angle
        _, min_ind = theta.min(dim=-1)   # shape: (B,)

        # 3) Gather the best quaternion from q_w_syms
        batch_indices = torch.arange(q_w_syms.size(0), device=q_w_syms.device)
        q_fz = q_w_syms[batch_indices, min_ind]

        # 4) Enforce q_fz[...,0] ≥ 0  (convention: keep scalar part positive)
        q_fz *= torch.sign(q_fz[..., :1])

        # 5) Reshape to desired output shape
        q_fz = q_fz.reshape(shape)
        return q_fz

def quat_angle(q1, q2, eps=1e-7):
    """
    Computes the angle between two unit quaternions q1 and q2.
    q1, q2: shape (..., 4)
    Returns: shape (...), the angle in radians.
    """
    # 1) Normalize the quaternions in case they drifted from unit length.
    q1 = q1 / (q1.norm(dim=-1, keepdim=True).clamp_min(eps))
    q2 = q2 / (q2.norm(dim=-1, keepdim=True).clamp_min(eps))

    # 2) Dot product: shape (...), since last dim is 4
    dot = (q1 * q2).sum(dim=-1)

    # 3) The absolute value handles the q <-> -q symmetry
    dot_clamped = dot.abs().clamp(max=1.0)

    # 4) Angle = 2 * arccos( |dot| )
    angle = 2.0 * torch.arccos(dot_clamped)
    return angle

def find_symmetry(qfz, qtarget, syms):
    """
    For each quaternion in qfz, find which symmetry in 'syms' yields the minimal 
    misorientation angle to qtarget.

    qfz:      shape (..., 4)    e.g. [B, 4]
    qtarget:  shape (..., 4) or (4); must be broadcastable with qfz
    syms:     shape (N, 4)      N symmetry quaternions
    returns:  shape (..., 4)    the symmetry in 'syms' that best aligns qfz to qtarget
    """
    # Flatten qfz to [B, 4] if it has extra leading dims
    original_shape = qfz.shape  # e.g. [B, 4]
    qfz_flat = qfz.reshape(-1, 4)  # [B, 4]

    # 'outer_prod(qfz_flat, syms)' should produce [B, N, 4]
    # each row i in [B] enumerates qfz_flat[i] * each syms[j]
    q_w_syms = outer_prod(qfz_flat, syms)  # shape [B, N, 4]

    # Similarly flatten qtarget if it has matching leading dims. If qtarget is just (4,) 
    # we can treat it as a single quaternion for all qfz. If qtarget has shape [B, 4], 
    # it must match qfz_flat's leading dimension.
    if qtarget.dim() == 1:
        # shape (4,) => single quaternion for all
        # Expand to [1, 1, 4] so it can broadcast with [B, N, 4]
        qtarget_expanded = qtarget.view(1, 1, 4)
    else:
        # Suppose qtarget also has shape (..., 4) => flatten:
        qtarget_flat = qtarget.reshape(-1, 4)  # [B, 4] if it matches qfz_flat
        # Expand to [B, 1, 4] so it can broadcast with [B, N, 4]
        qtarget_expanded = qtarget_flat.unsqueeze(1)

    # Vectorized angle computation => shape [B, N]
    # 'quat_angle' should accept two tensors of shape [B, N, 4], returning [B, N].
    angles = quat_angle(q_w_syms, qtarget_expanded)  # [B, N]

    # Find index of minimal angle for each batch element => shape [B]
    _, min_ind = angles.min(dim=-1)

    # Gather the corresponding symmetry => shape [B, 4]
    # syms[min_ind] won't work directly because min_ind is [B] but 'syms' is [N,4]
    # We can do it by advanced indexing:
    syms_min = syms[min_ind]  # shape [B, 4]

    # Finally, reshape back to original leading dims (if qfz had shape [B, 4], no change).
    syms_min = syms_min.view(original_shape[:-1] + (4,))

    return syms_min


def rev_map_quat(qfz, q, syms):
        shape = q.shape
        q = q.reshape((-1,4))
        syms = syms.cuda()
        qfz_w_syms = outer_prod(qfz, syms)
        
        # get the transfomration vector to go from q1 to q2




def scalar_first2last(X):
        return torch.roll(X,-1,-1)

def scalar_last2first(X):
        return torch.roll(X,1,-1)

def conj(q):
        q_out = q.clone()
        q_out[...,1:] *= -1
        return q_out


def rotate(q,points,element_wise=False):
        points = torch.as_tensor(points)
        P = torch.zeros(points.shape[:-1] + (4,),dtype=q.dtype,device=q.device)
        assert points.shape[-1] == 3, 'Last dimension must be of size 3'
        P[...,1:] = points
        if element_wise:
                X_int = hadamard_prod(q,P)
                X_out = hadamard_prod(X_int,conj(q))
        else:
                X_int = outer_prod(q,P)
                inds = (slice(None),)*(len(q.shape)-1) + \
                                (None,)*(len(P.shape)) + (slice(None),)
                X_out = (vec2mat(X_int) * conj(q)[inds]).sum(-1)
        return X_out[...,1:]


def linear_interpolation(q1, q2, frac):
    #inner product
    # scalar last convention
    lamda = torch.inner(q1, q2)
    mask = (lamda < 0)
    mask_ind = mask.nonzero()
    q2[mask_ind] = -q2[mask_ind]
    
    qin = q1 + frac * (q2 - q1)
    qin /= torch.norm(qin, dim=-1, keepdim=True) 
    
    return qin

def slerp_ebsd2d(input, scale_factor_h, scale_factor_w):
    import pdb; pdb.set_trace()
    source_H = input.shape[0]
    source_W = input.shape[1]
    source_C = input.shape[2]

    resized_H = int(source_H * scale_factor_h)
    resized_W = int(source_W * scale_factor_w)  
    
    output = np.zeros((resized_H, resized_W, source_C))
   
    def read_pixel(x, y):
        x = np.clip(x, 0, source_W - 1)
        y = np.clip(y, 0, source_H - 1)
        return input[y, x]   

    def quat_interpolate(x, y):
        #import pdb; pdb.set_trace()
        x1 = int(np.floor(x))
        x2 = x1 + 1

        y1 = int(np.floor(y))
        y2 = y1 + 1

        P11 = read_pixel(x1, y1)
        P12 = read_pixel(x1, y2)
        P21 = read_pixel(x2, y1)
        P22 = read_pixel(x2, y2)
        
        #return (P11 * (x2 - x) * (y2 - y) + 
        #        P12 * (x2 - x) * (y - y1) + 
        #        P21 * (x - x1) * (y2 - y) + 
        #        P22 * (x - x1) * (y - y1)) / ((x2 - x1) * (y2 - y1))
   
        P_int = slerp_quat(slerp_quat(P11, P12, 0.5), slerp_quat(P21, P22, 0.5), 0.5)
        #P_int =0.25 * slerp_quat(P11,P12, 0.5) + 0.25 * slerp_quat(P11,P21, 0.5) + 0.25 * slerp_quat(P21,P22, 0.5) + 0.25 * slerp_quat(P12,P22, 0.5)

        #import pdb; pdb.set_trace()
        return P_int

    for dst_y in range(resized_H):
        for dst_x in range(resized_W):
            #print(dst_y, dst_x)
            src_x = (dst_x + 0.5) / scale_factor_w - 0.5
            src_y = (dst_y + 0.5) / scale_factor_h - 0.5
            output[dst_y, dst_x] = quat_interpolate(src_x, src_y)

    import pdb; pdb.set_trace()
    return output    

def slerp_quat(q0, q1, t):
    #import pdb; pdb.set_trace()
        
    epsilon = 0.0001  
    dot = np.sum(q0 * q1)

    if dot < 0:
        # quaternions are poinitng in opposite directions 
        # use equivalent alternative representation for q2
        q1 = -q1
     
    if np.absolute(1 - dot) < epsilon:   
        # quaternions are nearly parallel
        # linear interpolation
        qin = q0 + t*(q1 - q0)
    else:
        # spherical interpolation
        dot = np.clip(dot, -1.0, 1.0)
        omega = np.arccos(dot)
        so = np.sin(omega)
        qin = np.sin((1.0-t)*omega) / so * q0 + np.sin(t*omega)/so * q1

    #norm = np.linalg.norm(qin)
    #qin_norm = qin / norm

    return qin




def slerp_quat_multidimensional(q0, q1, t):
    import pdb; pdb.set_trace()
    q0 = torch.tensor(q0)
    q1 = torch.tensor(q1)
    
    epsilon = 0.0001  
    dot = torch.sum(q0 * q1, dim=-1, keepdim=True)

    mask_ops = dot < 0
    # quaternions are poinitng in opposite directions 
    # use equivalent alternative representation for q2 
    mask_ind_ops = mask_ops.nonzero()
    q1[mask_ind_ops] = -q1[mask_ind_ops]

    # quaternions are nearly parallel
    # linear interpolation
    mask_prll = torch.norm(1 - dot, dim=-1, keepdim=True) < epsilon
    mask_prll = (mask_prll).float() 
    qin_maskprll = q0 + t*(q1 - q0)
   
    # spherical interpolation
    dot = torch.clamp(dot, -1.0, 1.0)
    omega = torch.acos(dot)
    so = torch.sin(omega)
    qin_mask_noprll = torch.sin((1.0-t)*omega) / so * q0 + torch.sin(t*omega)/so * q1

    qin = mask_prll * qin_maskprll + (1-mask_prll) * qin_mask_noprll

    norm = torch.norm(qin, dim=-1, keepdim=True)
    qin_norm = qin / norm

    return qin_norm

def save_quat(arr_list):
    #import pdb; pdb.set_trace()
    postfix = ['LR', 'Slerp', 'HR']
    kwargs_imshow = {'vmin': -1, 'vmax': 1}
    channels = ['q1', 'q2', 'q3', 'q0']

    for ch_num, channel in enumerate(channels):
        fig, axes = plt.subplots((len(postfix)+1)//3,3, figsize=(14,12), constrained_layout = True)
        for ax, arr, title in zip(axes.reshape(-1), arr_list, postfix):
            if ch_num == 0:
                np.save(f'slerp_out_{title}.npy', arr)
                
            img_numpy = arr[:,:, ch_num]
            im = ax.imshow(img_numpy, **kwargs_imshow, cmap='jet')
            ax.set_title(title, fontweight="bold")

        cbar =fig.colorbar(im, ax = axes.ravel().tolist(), shrink=0.95)
        cbar.set_ticklabels(np.arange(0,1,0.2))
        cbar.set_ticklabels([-1 , 0, 1])
    
        plt.savefig(f'slerp_out_{channel}.png')
        
        plt.close()


def normalize(x):
        x_norm = torch.norm(x, dim=-1, keepdim=True)
                # make ||q|| = 1
        y_norm = torch.div(x, x_norm) 

        return y_norm
     

def post_process(x, org_shape):
    #import pdb; pdb.set_trace()
    #rotate by 0 or 180 degrees about x axis
    hcp_r1 = torch.eye(4)[:2]

    # rotate about 0, 60, ... 300 degrees about z axis
    hcp_r2 = torch.zeros((6,4))
    hcp_r2[:,0] = torch.cos(torch.arange(6)/6*np.pi)
    hcp_r2[:,3] = torch.sin(torch.arange(6)/6*np.pi)
    hcp_syms = outer_prod(hcp_r1,hcp_r2).reshape((-1,4))
    hcp_syms = hcp_syms.type(torch.DoubleTensor)

    x = torch.tensor(x)
    x = x.cuda()
    x = normalize(x)
    h, w, ch = org_shape
   
    #import pdb; pdb.set_trace() 
    x = scalar_last2first(x)
    x = fz_reduce(x, hcp_syms)
    
    x = scalar_first2last(x)
    x = x.detach().cpu().numpy()
   
    return x
    


# A simple script to test the quats class for numpy and torch
if __name__ == '__main__':

        import pdb; pdb.set_trace()
        np.random.seed(1)
        N = 6
        M = 6
        K = 13

        def test1(dtype,device):

                q1 = rand_quats(M,dtype).to(device)
                q2 = rand_quats(N,dtype).to(device)
                q3 = rand_quats(M,dtype).to(device)
                p1 = rand_points(K,dtype).to(device)

                p2 = rotate(q2,rotate(q1,p1))
                p3 = rotate(outer_prod(q2,q1),p1)
                p4 = rotate(conj(q1[:,None]),rotate(q1,p1),element_wise=True)

                print('Composition of rotation error:')
                err = abs(p2-p3).sum()/len(p2.reshape(-1))
                print('\t',err)

                print('Rotate then apply inverse rotation error:')
                err = abs(p4-p1).sum()/len(p1.reshape(-1))
                print('\t',err,'\n')

       
        def test2(dtype,device):
                import pdb; pdb.set_trace()
                q1 = rand_quats(M,dtype).to(device)
                q2 = rand_quats(N,dtype).to(device)
                #q_in = linear_interpolation(q1, q2, 0.5)
                q_in = slerp(q1, q2, 0.5)
                
                
        def test3(dtype, device):
            import pdb;pdb.set_trace()
            img = np.load(f'Ti64_orthogonal_sectioning_Test_X_Block_lr_0.npy')
            img_hr = np.load(f'Ti64_orthogonal_sectioning_Test_X_Block_hr_0.npy')
            scale_factor = 2
            img_upsample = slerp_ebsd2d(img, scale_factor, scale_factor)
            img_upsample = post_process(img_upsample, img_hr.shape)
            save_quat([img, img_upsample, img_hr]) 
            import pdb; pdb.set_trace() 
 
        print('CPU Float 32')
        test3(torch.cuda.FloatTensor,'cpu')

        print('CPU Float64')
        test2(torch.cuda.DoubleTensor,'cpu')     

        if torch.cuda.is_available():

                print('CUDA Float 32')
                test2(torch.cuda.FloatTensor,'cuda')

                print('CUDA Float64')
                test2(torch.cuda.DoubleTensor,'cuda') 

        else:
                print('No CUDA')

