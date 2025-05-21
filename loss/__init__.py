import os
from importlib import import_module

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

class Loss(nn.modules.loss._Loss):
    def __init__(self, args, ckp):
        super(Loss, self).__init__()
        print('Preparing loss function:')
        
        self.include_consistency_loss = args.include_consistency_loss
        self.n_GPUs = args.n_GPUs
        self.loss = []
        self.loss_module = nn.ModuleList()
        for loss in args.loss.split('+'):
            weight, loss_type = loss.split('*')
            if loss_type == 'MSE':
                loss_function = nn.MSELoss()
            elif loss_type == 'L1':
                loss_function = nn.L1Loss()
            elif loss_type == 'L1_Charb':
                module = import_module('loss.L1_Charbonnier')
                loss_function = getattr(module, 'L1_Charbonnier')()
            elif loss_type == 'gradient':
                module = import_module('loss.gradient_loss')
                loss_function = getattr(module, 'gradient_loss')()
            elif loss_type == 'MisOrientation_EdgeLoss':
                module = import_module('loss.MisOrientation_EdgeLoss')
                loss_function = getattr(module, 'MisOrientation_EdgeLoss')(args)
            elif loss_type.find('VGG') >= 0:
                module = import_module('loss.vgg')
                loss_function = getattr(module, 'VGG')(
                    loss_type[3:],
                    rgb_range=args.rgb_range
                )
            elif loss_type == 'GAN':
                module = import_module('loss.adversarial')
                loss_function = getattr(module, 'Adversarial')(
                    args,
                    loss_type
                )
            elif loss_type == 'GAN_Symm':
                module = import_module('loss.adversarial_with_symm')
                loss_function = getattr(module, 'Adversarial_with_symm')(
                    args,
                    loss_type
                )

            elif loss_type.find('MisOrientation') >= 0:
                module = import_module('loss.misorientation')
                loss_function = getattr(module, 'MisOrientation')(args, mode=True)                 
            
            self.loss.append({
                'type': loss_type,
                'weight': float(weight),
                'function': loss_function}
            )
            #if loss_type.find('GAN') >= 0:
            #    self.loss.append({'type': 'DIS', 'weight': 1, 'function': None})

        if len(self.loss) > 1:
            self.loss.append({'type': 'Total', 'weight': 0, 'function': None})

        for l in self.loss:
            if l['function'] is not None:
                print('{:.3f} * {}'.format(l['weight'], l['type']))
                self.loss_module.append(l['function'])

        self.log = torch.Tensor()

        device = torch.device('cpu' if args.cpu else 'cuda')
        self.loss_module.to(device)
        if args.precision == 'half': self.loss_module.half()
        if not args.cpu and args.n_GPUs > 1:
            self.loss_module = nn.DataParallel(
                self.loss_module, range(args.n_GPUs)
            )

        if args.load != '.': self.load(ckp.dir, cpu=args.cpu)

    def forward(self, sr, hr):
        #import pdb; pdb.set_trace()
        losses = []
        cnt= 0
        summer=0
        for i, l in enumerate(self.loss):
            if l['function'] is not None:
                consistency_weight = 0
                if self.include_consistency_loss:
                    loss, consistency_loss = l['function'](sr, hr)
                    effective_loss = l['weight'] * loss + consistency_weight*l['weight'] * consistency_loss
                else:
                    loss = l['function'](sr, hr)
                    effective_loss = l['weight'] * loss
                losses.append(effective_loss)
                summer += effective_loss.item()
                cnt+= 1
                self.log[-1, i] = float(summer/ cnt)
            elif l['type'] == 'DIS':
                self.log[-1, i] += self.loss[i - 1]['function'].loss

        loss_sum = sum(losses)
        if len(self.loss) > 1:
            self.log[-1, -1] += loss_sum.item()

        return loss_sum

    def step(self):
        for l in self.get_loss_module():
            if hasattr(l, 'scheduler'):
                l.scheduler.step()

    def start_log(self):
        self.log = torch.cat((self.log, torch.zeros(1, len(self.loss))))

    def end_log(self, n_batches):
        self.log[-1].div_(n_batches)

    def display_loss(self, batch):
        n_samples = batch + 1
        log = []
        for l, c in zip(self.loss, self.log[-1]):
            if self.include_consistency_loss:
                log.append('[{} "with consitency loss": {:.4f}]'.format(l['type'], c / n_samples))
            else:
                log.append('[{}: {:.4f}]'.format(l['type'], c / n_samples))

        return ''.join(log)


    def plot_loss(self, apath, epoch):
        # Ensure epoch is an integer and at least 1
        epoch = max(1, int(epoch))

        # Ensure axis has at least one valid point
        axis = np.linspace(0, epoch - 1, epoch) if epoch > 1 else np.array([0])

        # Ensure the save directory exists
        os.makedirs(apath, exist_ok=True)

        # Check if self.loss is defined and contains data
        if not hasattr(self, 'loss') or not self.loss:
            print("Warning: No loss data available.")
            return

        for i, l in enumerate(self.loss):
            label = f"{l['type']}_Loss"
            #fig = plt.figure()
            plt.title(label)

            # Convert tensor safely
            try:
                loss_values = self.log[:, i]
                if isinstance(loss_values, torch.Tensor):
                    loss_values = loss_values.detach().cpu().numpy().astype(np.float64)
                loss_values = np.array(loss_values, dtype=np.float64).flatten()
            except IndexError:
                print(f"Error: Log does not contain enough entries for index {i}")
                continue

            # Handle cases where there's only 1 epoch
            if len(loss_values) == 1:
                plt.scatter(axis[0].astype(np.float64), float(loss_values[0]), label=label)  # Scatter for single points
            else:
                plt.plot(axis, loss_values, label=label)

            plt.legend()
            plt.xlabel("Epochs")
            plt.ylabel("Loss")
            plt.grid(True)

            # Save figure
            plt.savefig(f"{apath}/train_loss_{label}.pdf")
            #plt.close(fig)  # Free memory

        # Second plot (Rot Distance with Symm)
        label = "Rot Distance with Symm"
        fig = plt.figure()
        plt.title(label)

        # Ensure the last column exists in self.log
        try:
            loss_values = self.log[:, -1]
            if isinstance(loss_values, torch.Tensor):
                loss_values = loss_values.detach().cpu().numpy().astype(np.float32)
            loss_values = np.array(loss_values, dtype=np.float32).flatten()
        except IndexError:
            print("Error: Log does not contain enough entries for rotation distance loss.")
            return

        if len(loss_values) == 1:
            plt.scatter(axis, loss_values, label=label)
        else:
            plt.plot(axis, loss_values, label=label)

        plt.legend()
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.grid(True)

        # Save second plot
        plt.savefig(f"{apath}/loss_{label}.pdf")
        plt.close(fig)


    def get_loss_module(self):
        if self.n_GPUs == 1:
            return self.loss_module
        else:
            return self.loss_module.module

    def save(self, apath):
        torch.save(self.state_dict(), os.path.join(apath, 'loss.pt'))
        torch.save(self.log, os.path.join(apath, 'loss_log.pt'))

    def load(self, apath, cpu=False):
        if cpu:
            kwargs = {'map_location': lambda storage, loc: storage}
        else:
            kwargs = {}

        self.load_state_dict(torch.load(
            os.path.join(apath, 'loss.pt'),
            **kwargs
        ))
        self.log = torch.load(os.path.join(apath, 'loss_log.pt'))
        for l in self.loss_module:
            if hasattr(l, 'scheduler'):
                for _ in range(len(self.log)): l.scheduler.step()

