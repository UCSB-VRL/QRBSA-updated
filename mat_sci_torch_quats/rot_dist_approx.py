import torch

def derivative_of_euclid2rot(x):
    # derivative d/dx of arccos(1 - x/2) = -1 / sqrt(1 - (1 - x/2)^2) * (-1/2)
    # =>  1 / (2 * sqrt(1 - (1 - x/2)^2))
    return 1.0 / (2.0 * torch.sqrt(1 - (1 - x/2)**2 + 1e-9))

def euclid2rot(x):
    return torch.arccos(1 - 0.5*x**2)

class EuclidToRotApprox:
    def __init__(self,beta=0.1,eps=0.01):
        self.t = 2 - beta
        self.eps = eps
        t = torch.Tensor([self.t])

        slope_tensor = derivative_of_euclid2rot(torch.tensor([self.t], dtype=torch.float32))
        self.m = float(slope_tensor.item())  # convert to float
        #t.requires_grad = True
        y = euclid2rot(t)
        #y.backward()
        #self.m = float(t.grad)
        self.b = float(y) - self.m*self.t

    def __call__(self,x):
        x_clip = torch.clamp(x,self.eps,self.t)
        #mask = (x < self.beta).float()
        ## run experiment with 2*Y_ABS

        y_abs = torch.abs(x)
        y_lin = self.m*x + self.b
        y_rot = euclid2rot(x_clip)
        y_out = y_abs * (x <= self.eps).float() + \
                y_rot * torch.logical_and(x > self.eps,x < self.t).float() + \
                y_lin * (x >= self.t).float()

        return y_out


class RotDistLoss(torch.nn.Module):
    def forward(self, q_pred, q_gt):
        #import pdb; pdb.set_trace()
        q_pred_neg = torch.stack((q_pred,-q_pred), dim=-2)
        
        # check if q_gt has same dimension as q_pred
        if q_gt is not None:
            if q_pred.shape != q_gt.squeeze(-2).shape:
                q_gt = q_gt[...,None,:]
        
        # expand q_gt to the same shape as q_pred_neg
        q_gt = q_gt.expand_as(q_pred_neg)

        euclid_dist = torch.linalg.norm(q_pred_neg- q_gt.to(q_pred_neg.device), dim=-1)
        
        theta = 2*EuclidToRotApprox()(euclid_dist)
        #import pdb; pdb.set_trace()
        theta_min, selected_symmetries = theta.min(-1)

        return theta_min

        #Check if q_gt is not None and expand its shape to match q_pred


# class RotDistLoss(torch.nn.Module):
#     def forward(self, q_pred, q_gt):
#         # Check if q_gt is not None and expand its shape to match q_pred
#         if q_gt is not None:
#             # Squeeze q_gt to remove the singleton dimension
#             q_gt = q_gt.squeeze(dim=-2)  # Remove dimension with size 1 (e.g., [B, H, W, 1, 4] -> [B, H, W, 4])
        
#         # Now expand q_gt to the same shape as q_pred
#         q_gt = q_gt.expand_as(q_pred)  # Broadcasting q_gt to q_pred shape

#         # Compute the Euclidean distance between q_pred and q_gt (per quaternion in the batch)
#         euclid_dist = torch.linalg.norm(q_pred - q_gt.to(q_pred.device), dim=-1)

#         # Convert Euclidean distance to rotation angle (approximation)
#         theta = 2 * EuclidToRotApprox()(euclid_dist)
        
#         # Find the minimal angle (in case multiple symmetries or rotations are involved)
#         theta_min = theta.min(-1)[0]

#         return theta_min


