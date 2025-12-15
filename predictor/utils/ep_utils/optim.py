"""
Copyright (C) 2025 co-pace GmbH (a subsidiary of Continental AG).
Licensed under the BSD-3-Clause License.
@author: Yao Yue
"""

import math
from torch.optim.lr_scheduler import _LRScheduler


class WarmupCosineLR(_LRScheduler):
    """
    Warmup + Cosine Annealing Learning Rate Scheduler

    Arguments:
        optimizer (Optimizer): Wrapped optimizer.
        total_epochs (int): Total number of epochs for training.
        warmup_epochs (int): Number of epochs for linear warmup.
        min_lr (float): Final learning rate after cosine decay.
        max_lr (float): Initial learning rate after warmup.
        last_epoch (int): The index of last epoch. Default: -1.
        verbose (bool): Print information. Default: False.
    """
    
    def __init__(self, optimizer, total_epochs, warmup_epochs, min_lr, max_lr, last_epoch=-1, verbose=False):
        self.total_epochs = total_epochs
        self.warmup_epochs = warmup_epochs
        self.min_lr = min_lr
        self.max_lr = max_lr
        super(WarmupCosineLR, self).__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        # Compute the lr factor for current epoch
        if self.last_epoch < self.warmup_epochs:
            # Linear warmup
            factor = (self.last_epoch + 1) / self.warmup_epochs
            lr = self.max_lr * factor
        else:
            # Cosine annealing
            progress = (self.last_epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
            lr = self.min_lr + (self.max_lr - self.min_lr) * cosine_decay

        # Handle param groups scaling if any
        lrs = []
        for group in self.optimizer.param_groups:
            scale = group.get("lr_scale", 1.0)
            lrs.append(lr * scale)
        return lrs