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
from mat_sci_torch_quats.quats_old import fz_reduce, scalar_last2first, scalar_first2last
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
      - Q:    torch.Tensor of shape (N,4)
      - syms: torch.Tensor of shape (24,4) for FCC symmetry operators
    """
    # 1) Normalize input quaternions => shape (N,4)
    norms_Q = torch.norm(Q, dim=1, keepdim=True)
    Q_norm = Q / (norms_Q + 1e-12)

    # append -syms to the symmetry operators
    syms = torch.cat([syms, -syms], dim=0)  # shape => (48,4)

    # 2) Expand for broadcasting:
    #    syms => (1,24,4)
    #    Q_norm => (N,1,4)
    syms_ext = syms.unsqueeze(0)      # => shape (1,24,4)
    Q_ext = Q_norm.unsqueeze(1)       # => shape (N,1,4)


    # 3) Apply each FCC symmetry => shape (N,24,4)
    quat_sym = hamilton_product_torch(syms_ext, Q_ext)

    # 4) Normalize each symmetrical result => shape (N,24,4)
    norms_sym = torch.norm(quat_sym, dim=2, keepdim=True)
    quat_sym = quat_sym / (norms_sym + 1e-12)

    # 5) Dot with identity => real part is index 0
    #    => shape (N,24)
    dot_vals = quat_sym[..., 0].clamp_(-1.0, 1.0)

    # 6) Misorientation angles => 2 * arccos(real_part)
    angles = 2.0 * torch.acos(dot_vals)  # shape => (N,24)

    # 7) For each of the N quaternions, find the symmetry op giving min angle
    best_idx = torch.argmin(angles, dim=1)  # shape (N,)

    # Optionally check if min angle > 90 degrees
    best_angles = angles[torch.arange(angles.size(0)), best_idx]
    best_angles_deg = best_angles * 180.0 / math.pi
    if (best_angles_deg > 90).any():
        print("Warning: minimum angle > 90° found!")
        bad_idx = (best_angles_deg > 90).nonzero(as_tuple=True)[0]
        print(f"Bad indices: {bad_idx}")
        print(f"Bad angles (deg): {best_angles_deg[bad_idx]}")
        import pdb; pdb.set_trace()

    # 8) Gather best quaternions => shape (N,4)
    best_quat = quat_sym[torch.arange(Q.size(0)), best_idx, :]

    return best_quat

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
        self.T =10

        if self.args.load != '.':
            self.optimizer.load_state_dict(
                torch.load(os.path.join(ckp.dir, 'optimizer.pt'))
            )
            for _ in range(len(ckp.log)): self.scheduler.step
        self.error_last = 1e8
        self.epsilon = 0.001
        
        self.random_fz_quats= np.loadtxt('quaternions_fz.txt')

    def prepare_lr_transformed(self, lr, random_quats_conj):
        # **Apply random quaternion rotation to lr**
        B, C, H, W = lr.shape  # C=4
        # Step 1) Flatten each orientation map => shape (B,H*W,4)
        #   reorder to (B,HW,C), then reshape => (B,HW,4)
        lr_reshaped = lr.permute(0, 2, 3, 1).reshape(B, H*W, 4)

        # Step 2) Expand shapes for broadcasting:
        #   lr_reshaped => (B,HW,4) => (B,1,HW,4)
        #   quats_10 => (T,4) => (1,T,1,4)
        lr_reshaped = lr_reshaped[:, None, :, :]   # => shape (B,1,HW,4)
        random_quats_conj = random_quats_conj[None, :, None, :] # shape => (1,10,1,4) # => shape (10,4) => (1,10,1,4)

        lr_reshaped = scalar_last2first(lr_reshaped)
        # Step 3) Apply quaternion rotation
        # Hamilton product => (B,T,HW,4)
        out = hamilton_product_torch(random_quats_conj, lr_reshaped)
        # torch norm out dim=-1
        out = out / (torch.norm(out, dim=-1, keepdim=True) + 1e-8)
        out_reshape = out.squeeze(1).view(-1, 4)

        # Step 4) Reduce all to FZ => shape (B*T, HW, 4)
        out_fz = reduce_to_fz_fcc_all_torch(out_reshape, fcc_syms)
        out_fz = out_fz / (torch.norm(out_fz, dim=-1, keepdim=True) + 1e-8)
        out_fz= scalar_first2last(out_fz)

        # Step 5) Reshape back => (B*T,4,H,W)
        lr_transformed = out_fz.reshape(B*self.T, H, W, 4).permute(0, 3, 1, 2)
        return lr_transformed
        

    def prepare_hr_transformed(self, hr_org, random_quats_conj):

        # **Apply random quaternion rotation to hr**
        B, C, H, W = hr_org.shape  # C=4
        # Step 1) Flatten each orientation map => shape (B,4*H*W,4)
        #   reorder to (B,4HW,C), then reshape => (B,4HW,4)
        hr_reshaped = hr_org.permute(0, 2, 3, 1).reshape(B, H*W, 4)

        # Step 2) Expand shapes for broadcasting:
        #   lr_reshaped => (B,HW,4) => (B,1,HW,4)
        #   quats_10 => (T,4) => (1,T,1,4)
        hr_reshaped = hr_reshaped[:, None, :, :]   # => shape (B,1,HW,4)

        hr_reshaped = scalar_last2first(hr_reshaped)
        # Step 3) Apply quaternion rotation
        # Hamilton product => (B,T,HW,4)
        out = hamilton_product_torch(random_quats_conj, hr_reshaped)
        # torch norm out dim=-1
        out = out / (torch.norm(out, dim=-1, keepdim=True) + 1e-8)
        out_reshape = out.squeeze(1).view(-1, 4)

        # Step 4) Reduce all to FZ => shape (B*T, HW, 4)
        out_fz = reduce_to_fz_fcc_all_torch(out_reshape, fcc_syms)
        out_fz = out_fz / (torch.norm(out_fz, dim=-1, keepdim=True) + 1e-8)
        out_fz= scalar_first2last(out_fz)

        # Step 5) Reshape back => (B*T,4,H,W)
        hr_transformed = out_fz.reshape(B*T, H, W, 4).permute(0, 3, 1, 2)
        return hr_transformed

    def prepare_sr_transformed(self, sr, random_quats):
        
        B, C, H, W = sr.shape  # here B includes batch size and T, C=4
        # Scalar last to first for sr for transformation
        sr_transformed = scalar_last2first(sr)  # shape (B*T, H, W, 4)
        sr_transformed = hamilton_product_torch(random_quats, sr_transformed.permute(0, 2, 3, 1).view(B,-1, 4))
        sr_transformed = sr_transformed.squeeze(1).view(-1, 4)
        sr_transformed = sr_transformed / (torch.norm(sr_transformed, dim=-1, keepdim=True) + 1e-8)

        # tranform sr_transformed to FZ
        sr_transformed = reduce_to_fz_fcc_all_torch(sr_transformed, fcc_syms)
        sr_transformed = sr_transformed / (torch.norm(sr_transformed, dim=1, keepdim=True) + 1e-8)

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
            if self.args.prog_patch:
                lr, hr = common.get_prog_patch_1D(hr, epoch, self.args.scale) 
        
            B, C, H, W = lr.shape  # C=4
            # pull 10 random quaternions from self.random_fz_quats
            #import pdb; pdb.set_trace()
            random_indices = np.random.choice(self.random_fz_quats.shape[0], self.T, replace=False)
            random_quats = self.random_fz_quats[random_indices]
            random_quats = torch.tensor(random_quats, dtype=torch.float32)

            random_quats_conj= random_quats.clone()
            random_quats_conj[:, 1:] *= -1
            random_quats_conj = random_quats_conj / (torch.norm(random_quats_conj, dim=1, keepdim=True) + 1e-8)

            # GET LR_TRANSFORMED from LR, HR_TRANSFORMED from HR
            random_quats_conj= random_quats_conj[None, :,None, :].expand(B, T, -1, 4)
            lr_transformed= self.prepare_lr_transformed(lr, random_quats_conj)
            hr_transformed= self.prepare_hr_transformed(hr, random_quats_conj)

            timer_data.hold()
            timer_model.tic()

            # **Reset optimizer gradients before forward pass**
            for param in self.model.parameters():
                param.grad = None  # Ensures no stale gradients persist

            # **Forward pass**
            sr = self.model(lr_transformed, self.scale)
            # Normalize sr 
            sr = sr / (torch.norm(sr, dim=1, keepdim=True) + 1e-8)

            # GET SR_TRANSFORMED from SR
            random_quats = random_quats[:,None, :].expand(B*T, -1, 4)
            sr_transformed = self.prepare_sr_transformed(sr, random_quats)

            # ✅ Ensure `sr` has gradients
            sr_transformed.requires_grad_(True)

            ##############################################################################

            # **Compute loss safely**
            #### loss-1 ######
            hr = hr.repeat_interleave(T, dim=0) 
            if isinstance(sr, list):
                loss1 = torch.sum(torch.stack([self.loss(sr_transformed[j], hr) for j in range(len(sr_transformed))]))
            else:
                if self.args.include_consistency_loss:
                    #loss, consistency_loss = self.loss(sr, hr)
                    loss1 = self.loss(sr_transformed, hr)
                else:
                    loss1 = self.loss(sr_transformed, hr)  # ✅ Do not detach here!

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
            loss= loss1

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

            # for name, param in self.model.named_parameters():
            #     if param.grad is not None:
            #         max_grad = param.grad.abs().max().item()
            #         #print("max_grad:", max_grad)
            #         if max_grad > GRAD_EXPLOSION_THRESHOLD:
            #             import pdb; pdb.set_trace()
            #             print(f"⚠️ Warning: {name} has large gradients! Max grad: {max_grad:.4f}")

            # total_norm = 0.0
            # for param in self.model.parameters():
            #     if param.grad is not None:
            #         param_norm = param.grad.norm().item()
            #         total_norm += param_norm ** 2

            # total_norm = total_norm ** 0.5  # Compute total gradient norm

            # # Threshold for gradient explosion detection
            # GRAD_THRESHOLD = 10000

            # if total_norm > GRAD_THRESHOLD:
            #     print(f"Warning: Gradient norm too large ({total_norm:.2f})! Debugging...")
            #     import pdb;
            #     pdb.set_trace()  # Enter debug mode
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

        timer_model, timer_data = utility.timer(), utility.timer()
        self.model.eval()

        with torch.no_grad():
            total_val_loss = 0
            count = 0
            for batch, (lr, hr, filename_lr, filename_hr) in enumerate(self.loader_val):
                eval_acc = 0

                lr, hr = self.prepare([lr, hr])

                if self.args.prog_patch:
                    lr, hr = common.get_prog_patch_1D(hr, epoch, self.args.scale)

                sr = self.model(lr, self.scale)
                sr = sr / (torch.norm(sr, dim=1, keepdim=True) + 1e-8)
                _, ch, _, _ = sr.shape
                org_shape = hr.shape

                if isinstance(sr, list):
                    sr = self.post_process(sr[0], org_shape)
                else:
                    sr = self.post_process(sr, org_shape)

                lr = lr.permute(0, 2, 3, 1)
                hr = hr.permute(0, 2, 3, 1)

                B, C, H, W = lr.shape
                val_loss = self.mis_orient(sr, hr)
                val_loss = torch.mean(val_loss[0])
                val_loss = val_loss.detach().cpu().numpy()

                total_val_loss += val_loss
                count += 1
        
        avg_val_loss = total_val_loss / count
        print("avg Val loss:", avg_val_loss)

        self.total_val_loss_all.append(avg_val_loss)

        if avg_val_loss <= min(self.total_val_loss_all):
            self.ckp.save(self, epoch, is_best=True)

        self.ckp.plot_val_loss(self.total_val_loss_all, self.val_epochs_list)

        if self.args.save_results and (epoch % self.args.save_model_freq) == 0:
            print("--------------------Saving Model----------------------------")
            self.ckp.save(self, epoch)

        # Take the first sample by self.loader_val[0].
        for lr, hr, filename_lr, filename_hr in self.loader_val:
            # Stop after the first batch
            break

        # Ensure model is on the correct device (either GPU or CPU)
        device = next(self.model.parameters()).device  # This gets the device the model is on (e.g., cuda or cpu)

        # Move the inputs to the same device as the model (GPU or CPU)
        lr, hr = lr.to(device), hr.to(device)

        sr = self.model(lr, self.scale)

        # Process the super-resolved output
        if isinstance(sr, list):
            sr = self.post_process(sr[0], hr.shape)
        else:
            sr = self.post_process(sr, hr.shape)

        # Permute the tensor dimensions
        hr = hr.permute(0, 2, 3, 1)
        lr = lr.permute(0, 2, 3, 1)

        lr = lr.detach().cpu()
        hr = hr.detach().cpu()
        sr = sr.detach().cpu()

        # Now, pass the tensors to the save_results function
        save_list = [lr, hr, sr]
        modes = ['LR', 'HR', f'SR_{self.args.model}_{self.args.model_to_load}_{self.args.dist_type}']

        # Save results if required
        if self.args.save_results:
            self.ckp.save_results(filename_hr, save_list, modes, self.scale, epoch=self.args.model_to_load, dataset='Val')

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
                # After preparing sr, take median of 4 prepared quaternions
                random_indices = np.random.choice(self.random_fz_quats.shape[0], T, replace=False)
                random_quats = self.random_fz_quats[random_indices]
                random_quats = torch.tensor(random_quats, dtype=torch.float32)

                random_quats_conj= random_quats.clone()
                random_quats_conj[:, 1:] *= -1
                random_quats_conj = random_quats_conj / (torch.norm(random_quats_conj, dim=1, keepdim=True) + 1e-8)

                # GET LR_TRANSFORMED from LR, HR_TRANSFORMED from HR
                random_quats_conj= random_quats_conj[None, :,None, :].expand(B, T, -1, 4)
                lr_transformed= self.prepare_lr_transformed(lr, random_quats_conj)

                # **Forward pass**
                sr = self.model(lr_transformed, self.scale)
                # Normalize sr 
                sr = sr / (torch.norm(sr, dim=1, keepdim=True) + 1e-8)

                _, C,H,W = sr.shape 

                # FZ REDUCE BEFORE TAKING MEDIAN
                random_quats = random_quats[:,None, :].expand(B*T, -1, 4)
                sr_transformed = self.prepare_sr_transformed(sr, random_quats)

                # take the median pooling of sr along T dimension
                indices= torch.median(sr_transformed.view(B, T, 4, -1)[..., -1, :], dim=1)[1]
                indices = indices.unsqueeze(1).expand(-1, C, -1)
                
                # # take the mode pooling of sr along T dimension.
                mode_indices= torch.mode(sr_transformed.view(B, T, 4, -1)[..., -1, :], dim=1)[1] 
                mode_indices = mode_indices.unsqueeze(1).expand(-1, C, -1)

                sr_transformed = sr_transformed.reshape(B*T, C, -1)
                sr_transformed= torch.gather(sr_transformed, dim=0, index=mode_indices)  
                sr_transformed = sr_transformed / (torch.norm(sr_transformed, dim=1, keepdim=True) + 1e-8)
                sr= sr_transformed.reshape(B, C, H, W)

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


                save_list = [lr, hr, sr] + sr_up_trad
                #import pdb; pdb.set_trace()
                modes = ['LR', 'HR', f'SR_{self.args.model}_{self.args.model_to_load}_{self.args.dist_type}'] + modes   
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

    def upsample(mode, scale):
    
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


