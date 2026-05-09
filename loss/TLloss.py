import torch
from loss.lovasz import LovaszSoftmaxLoss
from loss.Tversky_loss import Tversky_loss


class TL_loss(torch.nn.Module):
    def __init__(self, alpha = 0.5, beta = 0.5, n_class= 2):
        super(TL_loss, self).__init__()
        self.alpha = alpha
        self.beta  = beta
        self.Tversky_loss = Tversky_loss(alpha = self.alpha, beta=self.beta, clsasses=n_class)
        self.lovasz_loss = LovaszSoftmaxLoss()

    def forward(self, outputs, labels):
        losm_loss = self.lovasz_loss(outputs, labels)
        tver_loss = self.Tversky_loss(outputs, labels)
        loss = 0.5*losm_loss + 0.5*tver_loss
        return loss

    def __call__(self, outputs, labels):
        return self.forward(outputs, labels)

