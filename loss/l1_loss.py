import torch.nn as nn


class l1_loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1_loss = nn.L1Loss()

    def forward(self, x, y):
        return self.l1_loss(x, y)
