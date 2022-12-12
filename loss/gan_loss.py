import torch
import torch.nn as nn
from l1_loss import l1_loss


class BCEWithLogits(nn.BCEWithLogitsLoss):
    def forward(self, pred_real, pred_fake=None):
        if pred_fake is not None:
            loss_real = super().forward(pred_real, torch.ones_like(pred_real))
            loss_fake = super().forward(pred_fake, torch.zeros_like(pred_fake))
            return loss_real + loss_fake
        else:
            loss = super().forward(pred_real, torch.ones_like(pred_real))
            return loss


class GAN_loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1_loss = l1_loss()
        self.gan_loss = BCEWithLogits()
