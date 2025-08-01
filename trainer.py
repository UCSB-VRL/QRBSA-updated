import os
import math
from decimal import Decimal
import utility
import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
from torch.nn import functional as F
from tqdm import tqdm
from collections import defaultdict 
from mat_sci_torch_quats.quats_old import fz_reduce, scalar_last2first, scalar_first2last, outer_prod
from mat_sci_torch_quats.symmetries import hcp_syms, fcc_syms  
from collections import defaultdict
import time
from thop import profile
import common
import gc


def reduce_to_fz_fcc_all_torch(Q, syms):
    """
    For all quaternions in Q (shape = (N,4)), find the best quaternion 
    in the FCC FZ (closest to the identity [1,0,0,0]).

    Returns best_quat (N,4).

    Arguments:
      - Q:    torch.Tensor of shape (N,4) ===> MAke sure that Q is (N, 4)
      - syms: torch.Tensor of shape (24,4) for FCC symmetry operators
    """

    # shape checks
    assert Q.ndim == 2 and Q.shape[1] == 4, \
        f"Q must be (N,4); got {tuple(Q.shape)}"
    assert syms.ndim == 2 and syms.shape == (24, 4), \
        f"syms must be (24,4); got {tuple(syms.shape)}"
    
    # 1) Normalize input quaternions => shape (N,4)
    norms_Q = torch.norm(Q, dim=-1, keepdim=True)
    Q_norm = Q / (norms_Q + 1e-12)

    # append -syms to the symmetry operators
    syms = torch.cat([syms, -syms], dim=0)  # shape => (48,4)

    # 2) Expand for broadcasting:
    #    syms => (1,24,4)
    #    Q_norm => (N,1,4)
    syms_ext = syms.unsqueeze(0)      # => shape (1,48,4)
    Q_ext = Q_norm.unsqueeze(1)       # => shape (N,1,4)

    # 3) Apply each FCC symmetry => shape (N,48,4)
    quat_sym = hamilton_product_torch(syms_ext, Q_ext)

    # get the indicies for the maximum for each quat_sym[..., 0]  # shape => (N,)
    idx = torch.argmax(quat_sym[..., 0], dim=-1)  # shape => (N,)
    idx_expanded_3D = idx.unsqueeze(-1).unsqueeze(-1).expand(-1, 1, 4)

    assert idx_expanded_3D.shape == (Q.shape[0], 1, 4), \
        f"idx_expanded_3D must be (N,1,4); got {tuple(idx_expanded_3D.shape)}"
    assert quat_sym.shape == (Q.shape[0], syms_ext.shape[1], 4), \
        f"quat_sym must be (N,48,4); got {tuple(quat_sym.shape)}"

    # 4) Gather best quaternions => shape (N,1,4)  
    best_quat = torch.gather(quat_sym, dim=1, index=idx_expanded_3D)  # shape => (N,4)

    return best_quat.reshape(Q.shape)  # shape => (N,4)
   
def hamilton_product_torch(q1, q2):
    """
    Pure PyTorch Hamilton product for quaternions.

    q1, q2: shape (..., 4)   [last dimension must be 4]
    Returns: shape (..., 4), broadcasted if necessary.

    Example shapes:
      - q1: (B, 1, 4)
      - q2: (1, T, 4)
      => result shape: (B, T, 4)
    """
    # Let PyTorch handle the broadcasting. We just do elementwise ops.
    # mount q1 on same device as q2
    q1 = q1.to(q2.device)
    # Decompose each quaternion into scalar + vector parts
    r1 = q1[..., 0]; x1 = q1[..., 1]; y1 = q1[..., 2]; z1 = q1[..., 3]
    r2 = q2[..., 0]; x2 = q2[..., 1]; y2 = q2[..., 2]; z2 = q2[..., 3]

    # Hamilton product formulas for each component
    ro = r1*r2 - x1*x2 - y1*y2 - z1*z2
    xo = r1*x2 + x1*r2 + y1*z2 - z1*y2
    yo = r1*y2 - x1*z2 + y1*r2 + z1*x2
    zo = r1*z2 + x1*y2 - y1*x2 + z1*r2

    # Stack them back along last dimension
    return torch.stack([ro, xo, yo, zo], dim=-1)

class Trainer():
    def __init__(self, args, loader_train, loader_val, loader_test, model, loss, ckp):
        self.args = args
        self.scale = args.scale
        self.ckp = ckp   # checkpoint
        self.loader_train = loader_train
        self.loader_val =  loader_val
        self.loader_test = loader_test
        self.model = model
        self.loss = loss
        self.epoch = args.current_epoch
        self.total_val_loss_all = []
        self.epoch_list = [] 
        self.val_epochs_list = []
        self.mis_orient = utility.Misorientation_dist(args)
        self.optimizer = utility.make_optimizer(args, self.model)
        #self.scheduler = utility.make_scheduler(args, self.optimizer)
        self.scheduler = utility.make_warmup_scheduler(args, self.optimizer)
        self.T = 1

        if self.args.load != '.':
            self.optimizer.load_state_dict(
                torch.load(os.path.join(ckp.dir, 'optimizer.pt'))
            )
            for _ in range(len(ckp.log)): self.scheduler.step
        self.error_last = 1e8
        self.epsilon = 0.001
        
        self.random_fz_quats= np.loadtxt('quaternions_edge_fz.txt')

    def prepare_lr_transformed(self, lr, random_quats_conj, syms_ext, T=None):
        if T is None:
            T = self.T
        # **Apply random quaternion rotation to lr**
        B, C, H, W = lr.shape  # C=4
        # Step 1) Flatten each orientation map => shape (B,H*W,4)
        #   reorder to (B,HW,C), then reshape => (B,HW,4)
        lr_reshaped = lr.permute(0, 2, 3, 1).reshape(B, H*W, 4)

        # Step 2) Expand shapes for broadcasting:
        #   lr_reshaped => (B,HW,4) => (B,1,HW,4)
        #   quats_10 => (T,4) => (1,T,1,4)
        lr_reshaped = lr_reshaped[:, None, :, :]   # => shape (B,1,HW,4)
        random_quats_conj = random_quats_conj[:, None, :] # shape => (1,10,1,4) # => shape (10,4) => (1,10,1,4)

        lr_reshaped = scalar_last2first(lr_reshaped)

        lr_reshaped= lr_reshaped[..., None, :]# shape => (B,48,1,4)
        #lr_48 = hamilton_product_torch(syms_ext, lr_reshaped)  # shape => (B,48,4)
        lr_48 = outer_prod(lr_reshaped.view(-1, 4), syms_ext.view(-1, 4))  # shape => (B,48,HW,4)

        random_quats_conj= random_quats_conj.clone()
        random_quats_conj = random_quats_conj.squeeze(1).view(B, T, 1, 4)
        assert random_quats_conj.shape == (B, T, 1, 4), \
            f"random_quats_conj must be (B,10,1,4); got {tuple(random_quats_conj.shape)}"
        assert lr_48.squeeze().reshape(B,1,-1,4).shape == (B,1, 48*H*W, 4), \
            f"lr_48 must be (B,1,48*HW,4); got {tuple(lr_48.squeeze().view(B,1,-1,4).shape)}"
        
        # trandform all 48 orientations with q*
        out_48=hamilton_product_torch(random_quats_conj, lr_48.squeeze().reshape(B,1,-1,4))
        out_48= out_48.reshape(B*T, H*W, -1, 4)

        # select the quaternions with the maximum value along dim=-2
        idx = torch.argmax(out_48[..., 0], dim=-1) # shape => (B, N,)
        idx_expanded_3D = idx.unsqueeze(-1).unsqueeze(-1).expand(B*T, -1, 1, 4)
        assert idx_expanded_3D.shape == (B*T, H*W, 1, 4), \
            f"idx_expanded_3D must be (B,HW,1,4); got {tuple(idx_expanded_3D.shape)}"
        assert out_48.shape == (B*T, H*W, syms_ext.shape[1], 4), \
            f"out_48 must be (B,HW,48,4); got {tuple(out_48.shape)}"
        
        # 4) Gather best quaternions => shape (B,HW,1,4)
        out = torch.gather(out_48, dim=2, index=idx_expanded_3D)  # shape => (B,HW,1,4)

        # torch norm out dim=-1
        out = out / (torch.norm(out, dim=-1, keepdim=True) + 1e-12)
        out_reshape = out.squeeze(1).view(-1, 4)

        out_fz = out_reshape
        # Step 4) Reduce all to FZ => shape (B*T, HW, 4)
        #out_fz = reduce_to_fz_fcc_all_torch(out_reshape, fcc_syms)
        #out_fz = out_fz / (torch.norm(out_fz, dim=-1, keepdim=True) + 1e-12)
        out_fz= scalar_first2last(out_fz)

        # Step 5) Reshape back => (B*T,4,H,W)
        lr_transformed = out_fz.reshape(B*T, H, W, 4).permute(0, 3, 1, 2)

        return lr_transformed
        
    def prepare_hr_transformed(self, hr, random_quats_conj, syms_ext):

        # **Apply random quaternion rotation to lr**
        B, C, H, W = hr.shape  # C=4
        # Step 1) Flatten each orientation map => shape (B,H*W,4)
        #   reorder to (B,HW,C), then reshape => (B,HW,4)
        hr_reshaped = hr.permute(0, 2, 3, 1).reshape(B, H*W, 4)

        # Step 2) Expand shapes for broadcasting:
        #   lr_reshaped => (B,HW,4) => (B,1,HW,4)
        #   quats_10 => (T,4) => (1,T,1,4)
        hr_reshaped = hr_reshaped[:, None, :, :]   # => shape (B,1,HW,4)
        random_quats_conj = random_quats_conj[:, None, :] # shape => (1,10,1,4) # => shape (10,4) => (1,10,1,4)

        hr_reshaped = scalar_last2first(hr_reshaped)

        hr_reshaped= hr_reshaped[..., None, :]# shape => (B,48,1,4)
        #hr_48 = hamilton_product_torch(syms_ext, hr_reshaped)  # shape => (B,48,4)
        hr_48 = outer_prod(hr_reshaped.view(-1, 4), syms_ext.view(-1, 4))  # shape => (B,48,HW,4)

        random_quats_conj= random_quats_conj.clone()
        random_quats_conj = random_quats_conj.squeeze(1).view(B, self.T, 1, 4)
        assert random_quats_conj.shape == (B, self.T, 1, 4), \
            f"random_quats_conj must be (B,10,1,4); got {tuple(random_quats_conj.shape)}"
        assert hr_48.squeeze().reshape(B,1,-1,4).shape == (B,1, 48*H*W, 4), \
            f"lr_48 must be (B,1,48*HW,4); got {tuple(hr_48.squeeze().reshape(B,1,-1,4).shape)}"
        
        # trandform all 48 orientations with q*
        out_48=hamilton_product_torch(random_quats_conj, hr_48.squeeze().reshape(B,1,-1,4))
        out_48= out_48.reshape(B*self.T, H*W, -1, 4)

        # select the quaternions with the maximum value along dim=-2
        idx = torch.argmax(out_48[..., 0], dim=-1) # shape => (B, N,)
        idx_expanded_3D = idx.unsqueeze(-1).unsqueeze(-1).expand(B*self.T, -1, 1, 4)
        assert idx_expanded_3D.shape == (B*self.T, H*W, 1, 4), \
            f"idx_expanded_3D must be (B,HW,1,4); got {tuple(idx_expanded_3D.shape)}"
        assert out_48.shape == (B*self.T, H*W, 48, 4), \
            f"out_48 must be (B,HW,48,4); got {tuple(out_48.shape)}"
        
        # 4) Gather best quaternions => shape (B,HW,1,4)
        out = torch.gather(out_48, dim=2, index=idx_expanded_3D)  # shape => (B,HW,1,4)

        # torch norm out dim=-1
        out = out / (torch.norm(out, dim=-1, keepdim=True) + 1e-12)
        out_reshape = out.squeeze(1).view(-1, 4)

        out_fz = out_reshape
        # Step 4) Reduce all to FZ => shape (B*T, HW, 4)
        #out_fz = reduce_to_fz_fcc_all_torch(out_reshape, fcc_syms)
        #out_fz = out_fz / (torch.norm(out_fz, dim=-1, keepdim=True) + 1e-12)
        out_fz= scalar_first2last(out_fz)

        # Step 5) Reshape back => (B*T,4,H,W)
        hr_transformed = out_fz.reshape(B*self.T, H, W, 4).permute(0, 3, 1, 2)

        return hr_transformed
        
    def prepare_sr_transformed(self, sr, random_quats, syms_ext):

        B, C, H, W = sr.shape  # here B includes batch size and T, C=4
        # Scalar last to first for sr for transformation
        sr_fn = sr.clone()
        sr_fn = scalar_last2first(sr_fn.permute(0,2,3,1))  # shape (B*T, H, W, 4)
        sr_transformed = hamilton_product_torch(random_quats, sr_fn.view(B,-1, 4))
        sr_transformed = sr_transformed.view(-1, 4)
        #sr_transformed = sr_transformed.view(-1, 1, 4)
        #sr_transformed = sr_transformed / (torch.norm(sr_transformed, dim=-1, keepdim=True) + 1e-12)

        # blow sr_transformed to 48 orientations
        # sr_transformed = hamilton_product_torch(syms_ext, sr_transformed)

        # # choose the idx with max scalar value
        # # select the quaternions with the maximum value along dim=-2
        # idx = torch.argmax(sr_transformed[..., 0], dim=-1).squeeze()  # shape => (B, N,)
        # idx_expanded_3D = idx.unsqueeze(-1).unsqueeze(-1).expand(-1, 1, 4)
        # assert idx_expanded_3D.shape == (B*H*W, 1, 4), \
        #     f"idx_expanded_3D must be (BHW,1,4); got {tuple(idx_expanded_3D.shape)}"
        # assert sr_transformed.shape == (B*H*W, 48, 4), \
        #     f"quat_sym must be (BHW,48,4); got {tuple(sr_transformed.shape)}"
        
        # # 4) Gather best quaternions => shape (B,HW,1,4)
        # sr_transformed = torch.gather(sr_transformed, dim=1, index=idx_expanded_3D)  # shape => (B,HW,1,4)
        # sr_transformed = sr_transformed / (torch.norm(sr_transformed, dim=-1, keepdim=True) + 1e-12)

        sr_transformed = fz_reduce(sr_transformed, fcc_syms)        

        # scalar first to last for sr_transformed
        sr_transformed = scalar_first2last(sr_transformed).view(B, H, W, 4).permute(0,3,1,2)  # shape (B*T, 4*H, W, 4)

        return sr_transformed

    def train(self): 
        self.optimizer.zero_grad(set_to_none=True)  # Ensure previous gradients are cleared
        
        self.loss.start_log()
        self.model.train()
        self.epoch+=1 
        epoch = self.epoch
        T=self.T
        timer_data, timer_model = utility.timer(), utility.timer()
        total_train_loss = 0

        for batch, (lr, hr, filename_lr, filename_hr) in enumerate(self.loader_train):
            
            learn_rate = self.scheduler.get_last_lr()[0]

            self.ckp.write_log(
                '[Epoch {}]\tLearning rate: {:.2e}'.format(epoch, Decimal(learn_rate))
            )

            lr, hr = self.prepare([lr, hr])
            # if self.args.prog_patch:
            #     lr, hr = common.get_prog_patch_1D(hr, epoch, self.args.scale) 
        
            B, C, H, W = lr.shape  # C=4
            #print(f"Batch {batch}: lr.shape={lr.shape}, hr.shape={hr.shape}, B={B}, C={C}, H={H}, W={W}")
            # pull 10 random quaternions from self.random_fz_quats
            #import pdb; pdb.set_trace()
            random_indices = np.random.choice(self.random_fz_quats.shape[0], self.T, replace=False)
            random_quats = self.random_fz_quats[random_indices]
            random_quats = torch.tensor(random_quats, dtype=torch.float32)

            random_quats = torch.tensor([1, 0, 0, 0], dtype=torch.float32).unsqueeze(0).expand(1, -1)
            random_quats_conj= random_quats.clone()
            random_quats_conj[:, 1:] *= -1
            random_quats_conj = random_quats_conj / (torch.norm(random_quats_conj, dim=1, keepdim=True) + 1e-12)

            # GET LR_TRANSFORMED from LR, HR_TRANSFORMED from HR
            random_quats_conj= random_quats_conj[None,:, None, :].expand(B, T, -1, 4)

             # symms expand lr_reshaped 
            syms_ext = torch.cat([fcc_syms, -fcc_syms], dim=0).unsqueeze(0)
            syms_ext=syms_ext.to(device="cuda:0") # shape => (B,48,4)

            lr_transformed= self.prepare_lr_transformed(lr, random_quats_conj, syms_ext=syms_ext)
            hr_transformed= self.prepare_hr_transformed(hr, random_quats_conj, syms_ext=syms_ext)

            timer_data.hold()
            timer_model.tic()

            # **Reset optimizer gradients before forward pass**
            for param in self.model.parameters():
                param.grad = None  # Ensures no stale gradients persist

            # **Forward pass**
            sr = self.model(lr_transformed, self.scale)
            # Normalize sr 
            sr = sr / (torch.norm(sr, dim=1, keepdim=True) + 1e-12)

            # if batch % 100 == 0:
            #     passed, max_err, errs = self.test_model_equivariance(lr_transformed)
            #     print("Equivariant:", passed)
            #     print("Max error:", max_err)
            #     print("Per-group errors:", errs)
                
            _, C,H,W = sr.shape 
            # GET SR_TRANSFORMED from SR
            random_quats = random_quats.repeat(B*T, 1).view(B*T, 1, 4)
            # sr_transformed = self.prepare_sr_transformed(sr, random_quats, syms_ext=syms_ext)
        
            # # ✅ Ensure `sr` has gradients
            # sr_transformed.requires_grad_(True)

            # #import pdb; pdb.set_trace()
            # ##############################################################################
            # # take the median pooling of sr along T dimension
            # median_indices= torch.median(sr_transformed.view(B, T, 4, -1)[..., -1, :], dim=1)[1]
            # median_indices = median_indices.unsqueeze(1).expand(-1, C, -1).reshape(B, 1, C, -1) 
            
            # # take the mode pooling of sr along T dimension.
            # mode_indices= torch.mode(sr_transformed.view(B, T, 4, -1)[..., -1, :], dim=1)[1] 
            # mode_indices = mode_indices.unsqueeze(1).expand(-1, C, -1).reshape(B, 1, C, -1) 
            
            # sr_transformed = sr_transformed.reshape(B,T, C, -1)
            # sr_transformed= torch.gather(sr_transformed, dim=1, index=median_indices)  


    
            # sr_transformed = sr_transformed / (torch.norm(sr_transformed, dim=2, keepdim=True) + 1e-12)
            # sr_transformed= sr_transformed.reshape(B, C, H, W)
            
            # # **Compute loss safely**
            # #### loss-1 ######

            # if isinstance(sr, list):
            #     loss1 = torch.sum(torch.stack([self.loss(sr_transformed[j], hr) for j in range(len(sr_transformed))]))
            # else:
            #     if self.args.include_consistency_loss:
            #         #loss, consistency_loss = self.loss(sr, hr)
            #         loss1 = self.loss(sr_transformed, hr)
            #     else:
            #         loss1 = self.loss(sr_transformed, hr)  # ✅ Do not detach here!

            loss1 = self.loss(sr, hr)

            # **Ensure loss is a scalar** 
            loss1 = loss1.mean()
            #if self.args.include_consistency_loss:
            #    consistency_loss=consistency_loss.mean()
            #    loss=loss+consistency_loss

            ###### loss-2 ######
            # **Compute loss safely**
            #### loss-1 ######
            if isinstance(sr, list):
                loss2 = torch.sum(torch.stack([self.loss(sr[j], hr) for j in range(len(sr))]))
            else:
                if self.args.include_consistency_loss:
                    #loss, consistency_loss = self.loss(sr, hr)
                    loss2 = self.loss(sr, hr_transformed)
                else:
                    loss2 = self.loss(sr, hr_transformed)  # ✅ Do not detach here!

            # **Ensure loss is a scalar** 
            loss2 = loss2.mean()
            #if self.args.include_consistency_loss:
            #    consistency_loss=consistency_loss.mean()
            #    loss=loss+consistency_loss

            # Choose which loss. 
            loss= loss2

            if batch % 5 == 0:
                print("Epoch:", epoch)
                print(f"Epoch {epoch}, Batch {batch}: loss.requires_grad={loss.requires_grad}, grad_fn={loss.grad_fn}")
                print("loss:", loss.item())

            # **Check for invalid loss values**
            if not torch.isfinite(loss):
                import pdb; pdb.set_trace()
                print(f'Skipping batch {batch + 1} due to NaN/Inf loss.')
                continue  # Skip this batch
            
            # **Ensure loss is within the allowed threshold**
            if loss.item() < self.args.skip_threshold * self.error_last:
                assert loss.grad_fn is not None, "❌ Loss is detached before backward!"  # Debugging check

            # If no explosion, backpropagate normally
            self.optimizer.zero_grad()  # Reset gradients
            #with torch.autograd.detect_anomaly():
            loss.backward() 

            # **Check for gradient explosion**  
            GRAD_EXPLOSION_THRESHOLD = 50  # Set a threshold for gradients
        
            self.optimizer.step()
                
            # ✅ Detach AFTER backpropagation
            loss = loss.detach()  # Detach loss from computation graph
            timer_model.hold()

            # **Logging**
            if (batch + 1) % self.args.print_every == 0:
                self.ckp.write_log('[{}/{}]\t{}\t{:.1f}+{:.1f}s'.format(
                    (batch + 1) * self.args.batch_size,
                    len(self.loader_train.dataset),
                    self.loss.display_loss(batch),
                    timer_model.release(),
                    timer_data.release()))

            timer_data.tic()

            # **Accumulate loss safely**
            if loss is not None:
                total_train_loss += loss.cpu().item()  # Fully remove from computation graph
        
        self.epoch_list.append(self.epoch)
        avg_train_loss = total_train_loss / (batch + 1)

        self.loss.end_log(len(self.loader_train))
        self.error_last = self.loss.log[-1, -1]

        self.scheduler.step()  # **Update learning rate scheduler**

    def val_error(self):
        epoch = self.epoch
        self.val_epochs_list.append(epoch)
        self.ckp.write_log('\nEvaluation:')
        T=self.T
        timer_model, timer_data = utility.timer(), utility.timer()
        self.model.eval()

        with torch.no_grad():
            total_val_loss = 0
            count = 0
            for batch, (lr, hr, filename_lr, filename_hr) in enumerate(self.loader_val):
                eval_acc = 0

                lr, hr = self.prepare([lr, hr])
                # if self.args.prog_patch:
                #     lr, hr = common.get_prog_patch_1D(hr, epoch, self.args.scale) 
            
                B, C, H, W = lr.shape  # C=4
               
                # pull 10 random quaternions from self.random_fz_quats
                #import pdb; pdb.set_trace()
                random_indices = np.random.choice(self.random_fz_quats.shape[0], self.T, replace=False)
                random_quats = self.random_fz_quats[random_indices]
                random_quats = torch.tensor(random_quats, dtype=torch.float32)

                random_quats = torch.tensor([1, 0, 0, 0], dtype=torch.float32).unsqueeze(0).expand(B, -1)
                random_quats_conj= random_quats.clone()
                random_quats_conj[:, 1:] *= -1
                random_quats_conj = random_quats_conj / (torch.norm(random_quats_conj, dim=1, keepdim=True) + 1e-12)

                # GET LR_TRANSFORMED from LR, HR_TRANSFORMED from HR
                random_quats_conj= random_quats_conj[None,:, None, :].expand(B, T, -1, 4)

                # symms expand lr_reshaped 
                syms_ext = torch.cat([fcc_syms, -fcc_syms], dim=0).unsqueeze(0)
                syms_ext=syms_ext.to(device="cuda:0") # shape => (B,48,4)

                lr_transformed= self.prepare_lr_transformed(lr, random_quats_conj, syms_ext=syms_ext)
                hr_transformed= self.prepare_hr_transformed(hr, random_quats_conj, syms_ext=syms_ext)

                timer_data.hold()
                timer_model.tic()

                # **Reset optimizer gradients before forward pass**
                for param in self.model.parameters():
                    param.grad = None  # Ensures no stale gradients persist

                # **Forward pass**
                sr = self.model(lr_transformed, self.scale)
                # Normalize sr 
                sr = sr / (torch.norm(sr, dim=1, keepdim=True) + 1e-12)
                # get the first sr == sr[0]
                sr_random_0 = sr[0].unsqueeze(0)

                _, C,H,W = hr.shape 
                # GET SR_TRANSFORMED from SR
                random_quats = random_quats[:,None, :].expand(B*T, -1, 4)
                # sr_transformed = self.prepare_sr_transformed(sr, random_quats, syms_ext=syms_ext)
                # sr_random_0_transformed = sr_transformed[0].unsqueeze(0)

                # # ✅ Ensure `sr` has gradients
                # sr_transformed.requires_grad_(True)

                # ##############################################################################
                # # take the median pooling of sr along T dimension
                # median_indices= torch.median(sr_transformed.view(B, T, 4, -1)[..., -1, :], dim=1)[1]
                # median_indices = median_indices.unsqueeze(1).expand(-1, C, -1).reshape(B, 1, C, -1) 
                
                # # take the mode pooling of sr along T dimension.
                # mode_indices= torch.mode(sr_transformed.view(B, T, 4, -1)[..., -1, :], dim=1)[1] 
                # mode_indices = mode_indices.unsqueeze(1).expand(-1, C, -1).reshape(B, 1, C, -1) 

                # sr_transformed = sr_transformed.reshape(B,T, C, -1)
                # sr_transformed= torch.gather(sr_transformed, dim=1, index=median_indices)  
                # sr_transformed = sr_transformed / (torch.norm(sr_transformed, dim=2, keepdim=True) + 1e-12)
                # sr_transformed= sr_transformed.reshape(B, C, H, W)
                # # **Compute loss safely**
                # #### loss-1 ######

                # if isinstance(sr, list):
                #     loss1 = torch.sum(torch.stack([self.loss(sr_transformed[j], hr) for j in range(len(sr_transformed))]))
                # else:
                #     if self.args.include_consistency_loss:
                #         #loss, consistency_loss = self.loss(sr, hr)
                #         loss1 = self.loss(sr_transformed, hr)
                #     else:
                #         loss1 = self.loss(sr_transformed, hr)  # ✅ Do not detach here!


                # Crop the sr to the same size as hr
                sr= sr[:, :, :H, :W]  # Crop to match hr size


                loss1 = self.loss(sr, hr)
                # **Ensure loss is a scalar** 
                val_loss= loss1
                val_loss = val_loss.detach().cpu().numpy()

                total_val_loss += val_loss
                count += 1
        
        passed, max_err, errs = self.test_model_equivariance(lr, scale=self.scale)
        print("Equivariant:", passed)
        print("Max error:", max_err)
        print("Per-group errors:", errs)

        sum_error = sum(errs)
        if sum_error > 0:
            print(f"Sum of equivariance errors: {sum_error:.6f}")
        else:
            print("No equivariance errors detected.")

        # EVALUATE EQUIVARIANCE ERRORS
        # Plot and update equivariance errors over epochs/batches
        save_dir = os.path.join(self.ckp.dir, 'equivariance_errors')
        os.makedirs(save_dir, exist_ok=True)
        errors_file = os.path.join(save_dir, 'equivariance_errors.npy')
        max_errors_file = os.path.join(save_dir, 'max_errors.npy')
        sum_errors_file = os.path.join(save_dir, 'sum_errors.npy')
        iteration_file = os.path.join(save_dir, 'iteration.npy')

        # Load previous errors if they exist, else initialize
        if os.path.exists(errors_file):
            all_errors = np.load(errors_file, allow_pickle=True).tolist()
        else:
            all_errors = []

        if os.path.exists(max_errors_file):
            max_errors = np.load(max_errors_file, allow_pickle=True).tolist()
        else:
            max_errors = []

        if os.path.exists(sum_errors_file):
            sum_errors = np.load(sum_errors_file, allow_pickle=True).tolist()
        else:
            sum_errors = []

        if os.path.exists(iteration_file):
            iteration = int(np.load(iteration_file))
        else:
            iteration = 0

        # Update errors
        all_errors.append(errs)
        max_errors.append(max(errs))
        sum_errors.append(sum(errs))
        iteration += 1

        # Save updated errors
        np.save(errors_file, np.array(all_errors, dtype=object))
        np.save(max_errors_file, np.array(max_errors, dtype=float))
        np.save(sum_errors_file, np.array(sum_errors, dtype=float))
        np.save(iteration_file, np.array(iteration, dtype=int))

        # Plot max and sum errors over iterations
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, iteration + 1), max_errors, label='Max Error', marker='o')
        plt.plot(range(1, iteration + 1), sum_errors, label='Sum of Errors', marker='x')
        plt.title('Equivariance Errors Over Iterations')
        plt.xlabel('Iteration')
        plt.ylabel('Error')
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'equivariance_errors_summary.png'))
        plt.close()

        # Ensure all_errors is a list of floats for plotting
        # Flatten nested lists if present
        def flatten(l):
            for item in l:
                if isinstance(item, list) or isinstance(item, np.ndarray):
                    yield from flatten(item)
                else:
                    yield item
                    
        all_errors_plot = [float(e) for e in flatten(all_errors)]
        plt.plot(all_errors_plot, marker='o')
        plt.title('Equivariance Error Over Iterations')
        plt.xlabel('Iteration')
        plt.ylabel('Error')
        plt.grid()
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'equivariance_errors_individual.png'))
        plt.close()

        ###################### PLOTS. #######################
        avg_val_loss = total_val_loss / count
        print("avg Val loss:", avg_val_loss)

        self.total_val_loss_all.append(avg_val_loss)

        if avg_val_loss <= min(self.total_val_loss_all):
            self.ckp.save(self, epoch, is_best=True)

        self.ckp.plot_val_loss(self.total_val_loss_all, self.val_epochs_list)

        if self.args.save_results and (epoch % self.args.save_model_freq) == 0:
            print("--------------------Saving Model----------------------------")
            self.ckp.save(self, epoch)

        lr= lr.permute(0,2,3,1).detach().cpu()
        lr_transformed_random_0 = lr_transformed[0].unsqueeze(0).permute(0,2,3,1).detach().cpu()
        hr= hr.permute(0,2,3,1).detach().cpu()

        # added to make this reynolds with aaugmentation work.
        sr_transformed =sr.clone()
        sr_random_0 = sr.clone()
        sr_random_0_transformed = sr.clone()

        sr_transformed= sr_transformed.permute(0,2,3,1).detach().cpu()
        sr_random_0 = sr_random_0.permute(0,2,3,1).detach().cpu()
        sr_random_0_transformed = sr_random_0_transformed.permute(0,2,3,1).detach().cpu()

        # Now, pass the tensors to the save_results function
        save_list = [lr, hr, lr_transformed_random_0, sr_random_0, sr_random_0_transformed, sr_transformed]
        modes = ['LR', 'HR', 'LR_Transformed_random0', f'SR_random0_{self.args.model}_{self.args.model_to_load}_{self.args.dist_type}'] +['SR_transformed_random0', 'SR_transformed']

        # Save results if required
        if self.args.save_results:
            self.ckp.save_results(filename_hr, save_list, modes, self.scale, epoch=self.args.model_to_load, dataset='Val')

    def test_model_equivariance(self, x, atol=1e-6, rtol=1e-5, scale=4):
        """
        Tests equivariance: f(g·x) ≈ g·f(x)
        for model with group_tensor (G,Cg,Cg).
        Input shape: (B,C,*spatial) with C % Cg == 0.
        """

        group_tensor = torch.tensor(np.load("./model/reynolds_utils/fcc_symmetry_group.npy"), dtype=torch.float32)
        #group_tensor_inv = torch.tensor(np.load("./model/reynolds_utils/fcc_symmetry_group_inv.npy"), dtype=torch.float32)

        self.model.eval()
        with torch.no_grad():
            B, C, *spatial = x.shape
            G, Cg, _ = group_tensor.shape
            assert C == Cg, f"Channels {C} must be same as group element {Cg}"

            # f(x)
            fx = self.model(x, 4)  # (B,Cout,*spatial_out)
            _, Cout, *spatial_out = fx.shape

            errors = []

            # put g on the same device as x
            group_tensor = group_tensor.to(x.device)  # (G,Cg,Cg)
            for g in group_tensor:  # (Cg,Cg)
                # g·x

                gx = torch.einsum("ci,bi...->bc...", g, x)  # (B,C,*spatial)
                f_gx = self.model(gx, scale)  # f(g·x)

                # g·f(x)
                g_fx = torch.einsum("ci,bi...->bc...", g, fx)  # (B,Cout,*spatial_out)

                # max error for this g
                diff = (f_gx - g_fx).abs().max().item()
                errors.append(diff)

            max_err = max(errors)
           
            passed = max_err < atol + rtol * fx.abs().max().item()
            return passed, max_err, errors

    def test(self, is_trad_results= False):
        #import pdb; pdb.set_trace()
        self.model.eval()     
        keys = [f'sr','bilinear', 'bicubic', 'nearest']
        total_psnr_dict = dict.fromkeys(keys,0)
        count = 0
        total_dist = 0
        
        with torch.no_grad():
            for batch, (lr, hr, filename_lr, filename_hr) in enumerate(self.loader_test):
                start_time = time.time()       
                print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
                print(f' LR Image: {filename_lr} and HR Image: {filename_hr}')
                print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
                
                modes = []
                sr_up_trad = []
                psnr_dict = defaultdict()
                lr, hr = self.prepare([lr, hr])
                #import pdb; pdb.set_trace() 
                sr = self.model(lr, self.scale)
                #sr = hr 
                org_shape = hr.shape
                                               
                #Interpolations
                if is_trad_results:
               
                    modes = ['bilinear', 'bicubic', 'nearest']
                    sr_up_trad = []
                    for mode in modes:
                        upsampling = nn.Upsample(scale_factor=self.scale, mode=mode)
                        sr_up = upsampling(lr)
                        sr_up = self.post_process(sr_up, org_shape)
                        sr_up_trad.append(sr_up)
            
                #import pdb; pdb.set_trace() 
                if isinstance(sr, list):
                    sr = self.post_process(sr[0], org_shape)
                else:
                    sr = self.post_process(sr, org_shape)
            
                #import pdb; pdb.set_trace() 
                hr = hr.permute(0,2,3,1)
                lr = lr.permute(0,2,3,1)
                #sr = sr.permute(0,2,3,1)
 
                save_list = [lr, hr, sr] + sr_up_trad
                #import pdb; pdb.set_trace()
                modes = ['LR', 'HR', f'SR_{self.args.model}_{self.args.model_to_load}_{self.args.dist_type}'] + modes   
                filenames = filename_hr
                
                if self.args.save_results:
                    self.ckp.save_results(filenames, save_list, modes, self.scale, epoch = self.args.model_to_load, dataset=self.args.test_dataset_type) 
                end_time = time.time()
                t = end_time - start_time
                print("Time:", t)

    def test_with_transformation(self, is_trad_results= False):
        #import pdb; pdb.set_trace()
        self.model.eval()     
        keys = [f'sr','bilinear', 'bicubic', 'nearest']
        total_psnr_dict = dict.fromkeys(keys,0)
        count = 0
        total_dist = 0
        # symms expand lr_reshaped 
        syms_ext = torch.cat([fcc_syms, -fcc_syms], dim=0).unsqueeze(0)
        syms_ext=syms_ext.to(device="cuda:0") # shape => (B,48,4)

        with torch.no_grad():
            for batch, (lr, hr, filename_lr, filename_hr) in enumerate(self.loader_test):
                
                start_time = time.time()       
                print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
                print(f' LR Image: {filename_lr} and HR Image: {filename_hr}')
                print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
                
                modes = []
                sr_up_trad = []
                psnr_dict = defaultdict()
                lr, hr = self.prepare([lr, hr])
                
                B,C,H,W = lr.shape  # C=4
                T=self.T
                sr = torch.zeros((B*T, C, 4*H, W), dtype=torch.float32).to(lr.device)
                                    #import pdb; pdb.set_trace()
                # After preparing sr, take median of 4 prepared quaternions
                random_indices = np.random.choice(self.random_fz_quats.shape[0], T, replace=False)
                random_quats = self.random_fz_quats[random_indices]
                random_quats = torch.tensor(random_quats, dtype=torch.float32)
                
                random_quats = torch.tensor([1, 0, 0, 0], dtype=torch.float32).unsqueeze(0).expand(B*T, -1)

                random_quats_conj= random_quats.clone()
                random_quats_conj[:, 1:] *= -1
                random_quats_conj = random_quats_conj / (torch.norm(random_quats_conj, dim=1, keepdim=True) + 1e-12)

                # GET LR_TRANSFORMED from LR, HR_TRANSFORMED from HR
                random_quats_conj= random_quats_conj[None, :,None, :].expand(B, T, -1, 4)
                it=0
                # while loop for self.T%50 
                while it < T/50:
                
                    lr_transformed= self.prepare_lr_transformed(lr, random_quats_conj[:, it*50:(it+1)*50, ...], syms_ext=syms_ext, T=50)
                    # **Forward pass**
                    #import pdb; pdb.set_trace()
                    sr[B*it*50:B*(it+1)*50, ...] =self.model(lr_transformed, self.scale)
                    # Normalize sr 
                    sr = sr / (torch.norm(sr, dim=1, keepdim=True) + 1e-12)
                    it+= 1
                    
                _, C,H,W = sr.shape 

                # FZ REDUCE BEFORE TAKING MEDIAN
                random_quats = random_quats[:,None, :].expand(B*T, -1, 4)                                                
                sr_transformed = self.prepare_sr_transformed(sr, random_quats, syms_ext=syms_ext)
                sr_transformed_random_0 = sr_transformed[0].unsqueeze(0).permute(0,2,3,1)
                sr_transformed_random_1 = sr_transformed[1].unsqueeze(0).permute(0,2,3,1)
                sr_transformed_random_2 = sr_transformed[2].unsqueeze(0).permute(0,2,3,1)

                # Remember sr is already in FZ reduced using prepare_sr_transformed
                if batch == 0:
                #    import pdb; pdb.set_trace()
                    
                    sr_clone = sr.clone()
                    sr_clone = scalar_last2first(sr_clone.permute(0,2,3,1))  # shape (B*T, H, W, 4)
                    sr_clone_transformed = hamilton_product_torch(random_quats, sr_clone.view(B*T,-1, 4))
                    sr_clone_transformed = sr_clone_transformed.view(-1, 4)
                    
                    temp = sr_clone_transformed.clone()
                    temp = fz_reduce(temp, fcc_syms)        
                    # scalar first to last for sr_transformed
                    temp = scalar_first2last(temp).view(B*T, H, W, 4).permute(0,3,1,2)  # shape (B*T, 4*H, W, 4)
                    sr_clone_transformed = scalar_first2last(sr_clone_transformed).view(B*T, H, W, 4).permute(0,3,1,2)  # shape (B*T, 4, H, W)

                    assert torch.allclose(temp, sr_transformed, atol=1e-6), "temp and sr_transformed are not equal"
                    
                    sr_save_list = []
                    sr_not_fz_reduced = []
                    for i in range(0, B*T):
                        sr_save_list.append(sr_transformed[i].unsqueeze(0).permute(0,2,3,1))
                        sr_not_fz_reduced.append(sr_clone_transformed[i].unsqueeze(0).permute(0,2,3,1))
                    sr_save = torch.cat(sr_save_list, dim=0)
                    sr_not_fz_reduced = torch.cat(sr_not_fz_reduced, dim=0)

                    # save sr_Save as npy
                    #import pdb; pdb.set_trace()
                    save_dir = os.path.join(self.ckp.dir, 'sr_save')
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)
                    save_path = os.path.join(
                        save_dir,
                        f'sr_save_{self.args.model}_{self.args.model_to_load}_{self.args.dist_type}.npy'
                    )
                    
                    np.save(save_path, sr_save.cpu().numpy())
                    save_path_not_fz_reduced = os.path.join(
                        save_dir,
                        f'sr_not_fz_reduced_{self.args.model}_{self.args.model_to_load}_{self.args.dist_type}.npy'
                    )
                    np.save(save_path_not_fz_reduced, sr_not_fz_reduced.cpu().numpy())

                # take the median pooling of sr along T dimension
                # Sort along T dimension before taking the median (ascending)
                sr_scalar = sr_transformed.view(B, T, 4, -1)[..., -1, :]
                median_indices = torch.median(sr_scalar, dim=1)[1]
                median_indices = median_indices.unsqueeze(1).expand(-1, C, -1)
                
                # # take the mode pooling of sr along T dimension.
                mode_indices= torch.mode(sr_scalar, dim=1)[1] 
                mode_indices = mode_indices.unsqueeze(1).expand(-1, C, -1)

                sr_transformed = sr_transformed.reshape(B*T, C, -1)
                sr_transformed= torch.gather(sr_transformed, dim=0, index=median_indices)  
                #sr_transformed= torch.gather(sr_transformed, dim=0, index=mode_indices)  

                sr_transformed = sr_transformed / (torch.norm(sr_transformed, dim=1, keepdim=True) + 1e-12)
                sr_transformed= sr_transformed.reshape(B, C, H, W)
                # sr = sr_transformed

                # save median indices and mode indices
                save_dir = os.path.join(self.ckp.dir, 'sr_save')
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                save_path_median = os.path.join(
                    save_dir,
                    f'sr_median_indices_{self.args.model}_{self.args.model_to_load}_{self.args.dist_type}.npy'
                )
                np.save(save_path_median, median_indices.reshape(B,C,H,W).cpu().numpy())

                #sr = hr 
                org_shape = hr.shape
                                             
                #Interpolations
                if is_trad_results:
               
                    modes = ['bilinear', 'bicubic', 'nearest']
                    sr_up_trad = []
                    for mode in modes:
                        upsampling = nn.Upsample(scale_factor=self.scale, mode=mode)
                        sr_up = upsampling(lr)
                        sr_up = self.post_process(sr_up, org_shape)
                        sr_up_trad.append(sr_up)

                #import pdb; pdb.set_trace() 
                if isinstance(sr, list):
                    sr = self.post_process(sr[0], org_shape)
                else:
                    sr = self.post_process(sr, org_shape)

                # prepare hr_transformed
                hr_transformed= self.prepare_hr_transformed(hr, random_quats_conj, syms_ext=syms_ext)
                B, C, H, W = hr_transformed.shape  # C=4

                # prepare hr_transformed_back
                hr_transformed_back =hr_transformed.clone()
                hr_transformed_back = scalar_last2first(hr_transformed_back.permute(0,2,3,1))
                hr_transformed_back = hamilton_product_torch(random_quats, hr_transformed_back.view(B, -1, 4))  # shape => (B*T, HW, 4)
                hr_transformed_back = hr_transformed_back.view(B, H*W, 4)  # shape => (B, HW, 4)
                # hr_transformed_back_48 = hamilton_product_torch(syms_ext, hr_transformed_back.view(-1,1,4))
                hr_transformed_back_48= outer_prod(hr_transformed_back.view(-1, 4), syms_ext.view(-1, 4))  # shape => (B,48,HW,4)
                # choose the idx with max scalar value
                idx = torch.argmax(hr_transformed_back_48[..., 0], dim=-1)
                idx_expanded_3D = idx.unsqueeze(-1).unsqueeze(-1).expand(-1, 1, 4)
                assert idx_expanded_3D.shape == (B*H*W, 1, 4), \
                    f"idx_expanded_3D must be (BHW,1,4); got {tuple(idx_expanded_3D.shape)}"
                assert hr_transformed_back_48.shape == (B*H*W, 48, 4), \
                    f"out_48 must be (B,HW,48,4); got {tuple(hr_transformed_back_48.shape)}"
                # 4) Gather best quaternions => shape (B,HW,1,4)
                hr_transformed_back = torch.gather(hr_transformed_back_48, dim=1, index=idx_expanded_3D)  # shape => (B,HW,1,4)
                # torch norm out dim=-1
                
                hr_transformed_back = hr_transformed_back / (torch.norm(hr_transformed_back, dim=-1, keepdim=True) + 1e-12)
                hr_transformed_back = hr_transformed_back.squeeze(1).reshape(-1, 4)

                hr_transformed_back = scalar_first2last(hr_transformed_back)
                # Step 5) Reshape back => (B*T,4,H,W)
                hr_transformed_back = hr_transformed_back.reshape(B, H, W, 4).permute(0,3,1,2)
 
                hr = hr.permute(0,2,3,1)
                lr = lr.permute(0,2,3,1)
                sr_0 = sr[0].unsqueeze(0)
                hr_transformed_0 = hr_transformed[0].unsqueeze(0).permute(0,2,3,1)
                hr_transformed_1 = hr_transformed[1].unsqueeze(0).permute(0,2,3,1)
                lr_transformed_0 = lr_transformed[0].unsqueeze(0).permute(0,2,3,1)
                sr_transformed = sr_transformed.permute(0,2,3,1)
                hr_transformed_back_0 = hr_transformed_back[0].unsqueeze(0).permute(0,2,3,1)
                hr_transformed_back_1 = hr_transformed_back[1].unsqueeze(0).permute(0,2,3,1)

                save_list = [lr, hr, sr_0] + sr_up_trad
                save_list.append(lr_transformed_0)
                save_list.append(hr_transformed_0)
                save_list.append(hr_transformed_1)
                save_list.append(sr_transformed_random_0)
                save_list.append(sr_transformed_random_1)
                save_list.append(sr_transformed_random_2)
                save_list.append(sr_transformed)
                save_list.append(hr_transformed_back_0)
                save_list.append(hr_transformed_back_1)
                
                #import pdb; pdb.set_trace()
                #modes = ['LR', 'HR', f'SR_{self.args.model}_{self.args.model_to_load}_{self.args.dist_type}'] + modes
                modes= ['LR', 'HR', f'SR_0_{self.args.model}_{self.args.model_to_load}_{self.args.dist_type}'] + modes + ['LR_transformed_0']+ ['HR_transformed0'] + ['HR_transformed1']+ ['SR_transformed_random_0'] + ['SR_transformed_random_1'] + ['SR_transformed_random_2'] + ['SR_transformed'] + ['HR_transformed_back_0'] + ['HR_transformed_back_1']
                filenames = filename_hr
                
                if self.args.save_results:
                    self.ckp.save_results(filenames, save_list, modes, self.scale, epoch = self.args.model_to_load, dataset=self.args.test_dataset_type) 
                end_time = time.time()
                t = end_time - start_time
                print("Time:", t)

    def post_process(self, x, org_shape):
        #import pdb; pdb.set_trace()
        b, ch, h, w = org_shape
        x = self.normalize(x)
        x = x[:,:,0:h,0:w]
        x = x.permute(0,2,3,1)

        # fz_reduction
        x = scalar_last2first(x)

        if self.args.syms_type == 'HCP':      
            x = fz_reduce(x, hcp_syms)
        elif self.args.syms_type == 'FCC':
            x = fz_reduce(x, fcc_syms)        

        x = scalar_first2last(x)
    
        return x

    def normalize(self,x):
        x_norm = torch.norm(x, dim=1, keepdim=True)
                # make ||q|| = 1
        y_norm = torch.div(x, x_norm) 

        return y_norm
                                   
    def prepare(self, l, volatile=False):
   
        device = torch.device('cpu' if self.args.cpu else 'cuda')
        def _prepare(tensor):
            if self.args.precision == 'half':
                tensor = tensor.half()
            return tensor.cuda()
           
        return [_prepare(_l) for _l in l]

    def upsample(self, mode, scale):
    
        upsampling = nn.Upsample(scale_factor=scale, mode=mode)
        sr_up = upsampling(lr)
        sr_up = self.normalize(sr_up)

        return sr_up

    def terminate(self):
        if self.args.test_only:
            self.test()
            return True
        else:
            #epoch = self.scheduler.last_epoch + 1
            epoch = self.epoch + 1
            return epoch >= self.args.epochs

    def is_val(self):
        epoch = self.epoch 
        if epoch % self.args.val_freq == 0:
            return True
        else:
            return False   


