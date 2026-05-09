import torch
import torch.nn.functional as F


class Tversky_loss(torch.nn.Module):
    def __init__(self, alpha = 0.5, beta = 0.5, clsasses = None):
        super(Tversky_loss, self).__init__()
        self.alpha = alpha
        self.beta  = beta
        self.classes = clsasses

    def forward(self, inputs, target, alpha=0.3, beta=0.7, smooth=1e-5):
        n, c, h, w = inputs.size()
        nt, ht, wt = target.size()
        target     = F.one_hot(target, num_classes=self.classes)
        temp_inputs = torch.softmax(inputs.transpose(1, 2).transpose(2, 3).contiguous().view(n, -1, c),
                                    -1)  # (1,65536,2) #softmax多分类?
        temp_target = target.view(n, -1, c)
        tp = torch.sum(temp_target[..., :] * temp_inputs, axis=[0, 1])
        fp = torch.sum(temp_inputs, axis=[0, 1]) - tp
        fn = torch.sum(temp_target[..., :], axis=[0, 1]) - tp
        tversky = (tp + smooth) / (tp + alpha * fp + beta * fn + smooth)
        loss = 1 - torch.mean(tversky)
        return loss

    def __call__(self, inputs, target):
        return self.forward(inputs, target)


