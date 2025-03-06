import torch
import torch.nn as nn


class HuberLoss(nn.Module):
    def __init__(self, delta=0.01):
        super(HuberLoss, self).__init__()
        self.delta = delta

    def __call__(self, in0, in1):
        mask = torch.zeros_like(in0)
        mann = torch.abs(in0 - in1)
        eucl = 0.5 * (mann**2)
        mask[...] = mann < self.delta

        loss = eucl * mask / self.delta + (mann - 0.5 * self.delta) * (1 - mask)
        return torch.sum(loss, dim=1, keepdim=True)
