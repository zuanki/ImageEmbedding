import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List


class CombinedLoss(nn.Module):
    def __init__(self,
                 losses: List[nn.Module]):
        super(CombinedLoss, self).__init__()
        self.losses = []
        for loss in losses:
            self.losses.append(loss)

    def forward(self, pred, mask):
        loss = 0
        for loss_fn in self.losses:
            loss += loss_fn(pred, mask)
        return loss


def vae_loss(recon_x, x, mean, logvar):
    # Reconstruction loss
    recon_loss = F.binary_cross_entropy(recon_x, x, reduction='sum')

    # KL divergence loss
    kl_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())

    return recon_loss + kl_loss


def vqvae2_loss(recon_x, x, mean, logvar, quantized, indices, beta=0.25):
    # Reconstruction loss
    recon_loss = F.binary_cross_entropy(recon_x, x, reduction='sum')

    # KL divergence loss
    kl_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())

    # Commitment loss
    commitment_loss = torch.mean(
        (quantized.detach() - mean) ** 2) + torch.mean((quantized - mean.detach()) ** 2)

    # Loss
    loss = recon_loss + kl_loss + beta * commitment_loss

    return loss
