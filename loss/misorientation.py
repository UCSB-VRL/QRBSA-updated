import torch
import torch.nn as nn
from mat_sci_torch_quats.losses import ActAndLoss, Loss, ConsitencyLoss
from mat_sci_torch_quats.symmetries import hcp_syms, fcc_syms

class MisOrientation(nn.Module):
    """Misoreintation loss."""
    def __init__(self, args, mode):
        super(MisOrientation, self).__init__()
        #import pdb; pdb.set_trace()
        dist_type = args.dist_type
        act = args.act_loss
        syms_req = args.syms_req
        syms_type = args.syms_type
        
        # hard coded for now
        #args.include_consistency_loss = False
        self.include_consistency_loss = args.include_consistency_loss
        
        print(f'Parameters for Training Loss')
        print('+++++++++++++++++++++++++++++++++++++++++')
        print(f'dist_type: {dist_type}  activation:{act}  Symmetry:{syms_req} Symmetry Type:{syms_type}')
        print('+++++++++++++++++++++++++++++++++++++++++++++++++')

        if syms_req:
            if syms_type == 'HCP':
                syms = hcp_syms
            elif syms_type == 'FCC':
                syms = fcc_syms
        else:
            syms = None
        self.act_loss = ActAndLoss(act,
                                    Loss(dist_func=dist_type, syms=syms),
                                    include_consistency_loss=self.include_consistency_loss,
                                    grain_consistency_loss=ConsitencyLoss(),
                                    quat_dim=1
            )

            
    def forward(self, sr, hr):
        #import pdb; pdb.set_trace()
        if not self.include_consistency_loss:
            loss = self.act_loss(sr, hr)
            loss = loss.mean()
            return loss
        else:
            loss, consistency_loss = self.act_loss(sr, hr)

            # check with mean and sum.
            loss=loss.mean()
            consistency_loss=consistency_loss.mean()
            return loss, consistency_loss
        return loss