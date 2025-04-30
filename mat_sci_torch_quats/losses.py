import torch
from mat_sci_torch_quats.quats import rand_quats, outer_prod, rot_dist, scalar_first2last,  scalar_last2first, validation_rot_dist_approx_MAT_symmetry, transformation_matrix_tensor, validation_min_angle_transformation
from mat_sci_torch_quats.symmetries import fcc_syms, hcp_syms
from mat_sci_torch_quats.rot_dist_approx import RotDistLoss
from mat_sci_torch_quats.quats import kernel_misorientation_deviation, fz_reduce, find_symmetry, matrix_hamilton_prod
from mat_sci_torch_quats.new_utils import sym_expand_transform
import torch.nn as nn
import torch.nn.functional as F


def l1(q1,q2):
        """ Basic L1 loss """
        return torch.mean(abs(q1-q2),dim=-1)

def l2(q1,q2):
        """ Basic L2 loss """
        return torch.sqrt(torch.mean((q1-q2)**2,dim=-1))


class Loss:
    def __init__(self, syms, mode=True):
        # import pdb; pdb.set_trace()
        self.syms = syms
        self.mode = mode
        
        self.min_angle_transform = MinimumAngleTransformation(self.mode)

    def __call__(self, qSR, qHR):
        # q1 is a multi-dimensional tensor of generated quaternions
        # q2 contains the respective ground-truth quaternions
        return self.min_angle_transform(qSR, qHR, self.syms.to(qSR.device))

class Loss:
        """ Wrapper for loss. Inclues option for symmetry as well """
        def __init__(self,dist_func,syms=None):
                # import pdb; pdb.set_trace()
                self.dist_type = dist_func
                if dist_func == 'l1':
                    self.dist_func = l1 
                elif dist_func == 'l2':
                    self.dist_func = l2
                elif dist_func == 'rot_dist':
                    self.dist_func = validation_min_angle_transformation # GETS CALLED DURING VALIDATION
                elif dist_func == 'rot_dist_approx_MAT_symmetry':
                  self.dist_func = RotDistLoss()
                elif dist_func == 'minimum_angle_transformation':
                #     import pdb; pdb.set_trace()
                    self.dist_func = RotDistLoss() # GETS CALLED DURING TRAINING
                elif dist_func == 'rot_dist_approx':
                    self.dist_func = RotDistLoss()
                elif dist_func == 'valid_symmHR_expand':
                    self.dist_func = RotDistLoss()
                elif dist_func == 'rot_dist_approx_without_symm':
                        self.dist_func = RotDistLoss()
                else:
                        print("no distance function was specified")
                
                self.syms = fcc_syms
                if syms is not None:
                        if (syms == 'fcc'):
                                self.syms = fcc_syms
                        elif (syms == 'hcp'):
                                self.syms = hcp_syms
                syms_neg = -self.syms
                self.syms = torch.cat((self.syms, syms_neg))
                self.syms = self.syms.cuda()
                #self.quat_dim = quat_dim

        def __call__(self,q1,q2):   

                if self.dist_type == 'minimum_angle_transformation':
                         ## Traiing with the minimum_angle_transformation based loss-function
                        T_min, selected_symmetries = transformation_matrix_tensor(q1, q2, self.syms)
                        zero_broadcast_tensor = torch.Tensor([1,0,0,0])

                        # broadcast the tensor to the same shape as T_min
                        zero_broadcast_tensor = zero_broadcast_tensor.reshape(1,1,1,4)
                        
                        dist_min= self.dist_func(T_min, zero_broadcast_tensor)
                        return dist_min, selected_symmetries

                elif self.dist_type == 'rot_dist_approx_without_symm':
                        if q2 is not None: q2 = q2[...,None,:]
                        dist = self.dist_func(q1,q2)
                        return (dist,())    
                        
                elif self.dist_type == 'rot_dist_approx':
                        q1_w_syms = outer_prod(q1,self.syms)
                        if q2 is not None: q2 = q2[...,None,:]
                        dists = self.dist_func(q1_w_syms,q2)
                        dist_min = dists.min(-1)[0]
                        return (dist_min, ())     
                                # T_series_min = rot_dist(q1, q2, self.syms)
                                # zero_broadcast_tensor = torch.Tensor([1,0,0,0])
                                # zero_broadcast_tensor = zero_broadcast_tensor.reshape(1,1,1,4) 
                                # return self.dist_func(T_min, zero_broadcast_tensor)

                elif self.dist_type == 'valid_symmHR_expand': 
                        # import pdb; pdb.set_trace()
                        
                        # Step 1: symmetry expand all q1 and reduce to qSRfz 
                        q1fz = fz_reduce(q1, self.syms)

                        # Step 2: take QHR to QHRfz
                        q2fz = fz_reduce(q2, self.syms)
                        
                        # Step 3: get the symmetry transformation T
                        chosen_sym = find_symmetry(q1fz, q1, self.syms)
                        
                        # Step 4: take q2fz to zone of q1 zone by applying the transformation T
                        q2_transformed = matrix_hamilton_prod(chosen_sym, q2fz)
                        
                        # Step 5: calculate the distance between q1 and q2_transformed
                        # import pdb; pdb.set_trace()
                        dists = self.dist_func(q1, q2_transformed)
                        #dist_min = dists.min(-1)[0]
                        return (dists, chosen_sym)

        def __str__(self):
                return f'Dist -> dist_func: {self.dist_func}, ' + \
                           f'syms: {self.syms is not None}'


class ConsitencyLoss:
        """ Wrapper for loss. Inclues option for symmetry as well """
        def __init__(self):
                self.dev_func = kernel_misorientation_deviation
                self.syms = fcc_syms
                # elif (syms == 'hcp'):
                #       self.syms = hcp_syms
                # else:
                #       print("no symmetry was specified")
                syms_neg = -self.syms
                self.syms = torch.cat((self.syms, syms_neg))
                
        def __call__(self, q1, q2, angles, selected_symmetries):
                return self.dev_func(q1, q2, angles, selected_symmetries, syms=self.syms)
               
        def __str__(self):
                return f'Dist -> dist_func: {self.dev_func}'

def tanhc(x):
        """
        Computes tanh(x)/x. For x close to 0, the function is defined, but not
        numerically stable. For values less than eps, a taylor series is used.
        """
        eps = 0.05
        mask = (torch.abs(x) < eps).float()
        # clip x values, to plug into tanh(x)/x
        x_clip = torch.clamp(abs(x),min=eps)
        # taylor series evaluation
        output_ts = 1 - (x**2)/3 + 2*(x**4)/15 - 17*(x**6)/315
        # regular function evaluation for tanh(x)/x
        output_ht = torch.tanh(x_clip)/x_clip
        # use taylor series if x is close to 0, otherwise, use tanh(x)/x
        output = mask*output_ts + (1-mask)*output_ht
        return output


def tanh_act(q):
        """ Scale a vector q such that ||q|| = tanh(||q||) """
        return q*tanhc(torch.norm(q,dim=-1,keepdim=True))
        
def safe_divide_act(q,eps=10**-5):
        """ Scale a vector such that ||q|| ~= 1 """
        return q/(eps+torch.norm(q,dim=-1,keepdim=True))


class ActAndLoss:
        """ Wraps together activation and loss """
        def __init__(self,act,loss, grain_consistency_loss, include_consistency_loss=False, quat_dim=-1):
                self.act = act
                self.loss = loss
                self.quat_dim = quat_dim
                self.include_consistency_loss = include_consistency_loss
                self.consistency_loss = grain_consistency_loss
        def __call__(self,X,labels):
                consistency_loss = 0
                # change to [b, ch, h, w] to [b, h, w, ch]
                X = torch.movedim(X,self.quat_dim,-1)
                labels = torch.movedim(labels,self.quat_dim,-1)
                # scalar first convention for outer product
                X = scalar_last2first(X)
                labels = scalar_last2first(labels)

                if self.act == 'tanhc':
                    X_act = tanh_act(X)
                elif self.act is None:
                    X_act = X 
                
                if not self.include_consistency_loss:
                        return self.loss(X_act, labels)
                else:
                        angles, selected_symmetries =self.loss(X_act,labels)
                        consistency_loss = self.consistency_loss(X_act,labels, angles, selected_symmetries)
                        return angles, consistency_loss
                
        def __str__(self):
                return f'Act and Loss: ({self.act},{self.loss})'


# A simple script to test the quats class for numpy and torch
if __name__ == '__main__':

        from symmetries import hcp_syms

        torch.manual_seed(1)
        
        q1 = torch.randn(7,4,17,19)
        q2 = torch.randn(7,4,17,19)

        q1 /= torch.norm(q1,dim=1,keepdim=True)

        q2.requires_grad = True

        acts_and_losses = list()
        
        for act in [None,tanh_act,safe_divide_act]:
                for syms in [None,hcp_syms]:
                        for dist in [l1,l2,rot_dist]:
                                acts_and_losses.append(ActAndLoss(act,Loss(dist,syms,1)))
        

        for i in acts_and_losses:
                print(i)
                d = i(q1,q2)
                L = d.sum()
                print(L)
