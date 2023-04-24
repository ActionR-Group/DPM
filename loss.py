import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.01):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.smoothing = smoothing

    def forward(self, x, target):
        confidence = 1. - self.smoothing
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


def loss_func(output, encoder_Z_distr, prior_Z_distr, p_y_given_z, temperature):
    soft = nn.Softmax(dim=1)
    I_ZX_bound = F.kl_div(input=soft(encoder_Z_distr.detach() / temperature),
                          target=soft(prior_Z_distr) / temperature)
    kl_loss = F.kl_div(input=soft(output.detach() / temperature),
                       target=soft(p_y_given_z) / temperature)
    return I_ZX_bound, kl_loss
