"""
Lovasz-Softmax and Jaccard hinge loss in PyTorch
Maxim Berman 2018 ESAT-PSI KU Leuven (MIT License)
"""

from __future__ import print_function, division
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np

try:
    from itertools import ifilterfalse
except ImportError:  # py3k
    from itertools import filterfalse as ifilterfalse


def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1. - intersection / union
    if p > 1:  # cover 1-pixel case
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def iou_binary(preds, labels, EMPTY=1., ignore=None, per_image=True):
    """
    IoU for foreground class
    binary: 1 foreground, 0 background
    """
    if not per_image:
        preds, labels = (preds,), (labels,)
    ious = []
    for pred, label in zip(preds, labels):
        intersection = ((label == 1) & (pred == 1)).sum()
        union = ((label == 1) | ((pred == 1) & (label != ignore))).sum()
        if not union:
            iou = EMPTY
        else:
            iou = float(intersection) / float(union)
        ious.append(iou)
    iou = mean(ious)  # mean accross images if per_image
    return 100 * iou


def iou(preds, labels, C, EMPTY=1., ignore=None, per_image=False):
    """
    Array of IoU for each (non ignored) class
    """
    if not per_image:
        preds, labels = (preds,), (labels,)
    ious = []
    for pred, label in zip(preds, labels):
        iou = []
        for i in range(C):
            if i != ignore:  # The ignored label is sometimes among predicted classes (ENet - CityScapes)
                intersection = ((label == i) & (pred == i)).sum()
                union = ((label == i) | ((pred == i) & (label != ignore))).sum()
                if not union:
                    iou.append(EMPTY)
                else:
                    iou.append(float(intersection) / float(union))
        ious.append(iou)
    ious = [mean(iou) for iou in zip(*ious)]  # mean accross images if per_image
    return 100 * np.array(ious)


# --------------------------- BINARY LOSSES ---------------------------


def lovasz_hinge(logits, labels, per_image=True, ignore=None):
    """
    Binary Lovasz hinge loss
      logits: [B, H, W] Variable, logits at each pixel (between -\infty and +\infty)
      labels: [B, H, W] Tensor, binary ground truth masks (0 or 1)
      per_image: compute the loss per image instead of per batch
      ignore: void class id
    """
    if per_image:
        loss = mean(lovasz_hinge_flat(*flatten_binary_scores(log.unsqueeze(0), lab.unsqueeze(0), ignore))
                    for log, lab in zip(logits, labels))
    else:
        loss = lovasz_hinge_flat(*flatten_binary_scores(logits, labels, ignore))
    return loss


def lovasz_hinge_flat(logits, labels):
    """
    Binary Lovasz hinge loss
      logits: [P] Variable, logits at each prediction (between -\infty and +\infty)
      labels: [P] Tensor, binary ground truth labels (0 or 1)
      ignore: label to ignore
    """
    if len(labels) == 0:
        # only void pixels, the gradients should be 0
        return logits.sum() * 0.
    signs = 2. * labels.float() - 1.
    errors = (1. - logits * Variable(signs))
    errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
    perm = perm.data
    gt_sorted = labels[perm]
    grad = lovasz_grad(gt_sorted)
    loss = torch.dot(F.relu(errors_sorted), Variable(grad))
    return loss


def flatten_binary_scores(scores, labels, ignore=None):
    """
    Flattens predictions in the batch (binary case)
    Remove labels equal to 'ignore'
    """
    scores = scores.view(-1)
    labels = labels.view(-1)
    if ignore is None:
        return scores, labels
    valid = (labels != ignore)
    vscores = scores[valid]
    vlabels = labels[valid]
    return vscores, vlabels


class StableBCELoss(torch.nn.modules.Module):
    def __init__(self):
        super(StableBCELoss, self).__init__()

    def forward(self, input, target):
        neg_abs = - input.abs()
        loss = input.clamp(min=0) - input * target + (1 + neg_abs.exp()).log()
        return loss.mean()


def binary_xloss(logits, labels, ignore=None):
    """
    Binary Cross entropy loss
      logits: [B, H, W] Variable, logits at each pixel (between -\infty and +\infty)
      labels: [B, H, W] Tensor, binary ground truth masks (0 or 1)
      ignore: void class id
    """
    logits, labels = flatten_binary_scores(logits, labels, ignore)
    loss = StableBCELoss()(logits, Variable(labels.float()))
    return loss


# --------------------------- MULTICLASS LOSSES ---------------------------


def lovasz_softmax(probas, labels, classes='present', per_image=False, ignore=None):
    """
    Multi-class Lovasz-Softmax loss
      probas: [B, C, H, W] Variable, class probabilities at each prediction (between 0 and 1).
              Interpreted as binary (sigmoid) output with outputs of size [B, H, W].
      labels: [B, H, W] Tensor, ground truth labels (between 0 and C - 1)
      classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
      per_image: compute the loss per image instead of per batch
      ignore: void class labels
    """
    if per_image:
        loss = mean(lovasz_softmax_flat(*flatten_probas(prob.unsqueeze(0), lab.unsqueeze(0), ignore), classes=classes)
                    for prob, lab in zip(probas, labels))
    else:
        loss = lovasz_softmax_flat(*flatten_probas(probas, labels, ignore), classes=classes)
    return loss


def lovasz_softmax_flat(probas, labels, classes='present'):
    """
    Multi-class Lovasz-Softmax loss
      probas: [P, C] Variable, class probabilities at each prediction (between 0 and 1)
      labels: [P] Tensor, ground truth labels (between 0 and C - 1)
      classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
    """
    if probas.numel() == 0:
        # only void pixels, the gradients should be 0
        return probas * 0.
    C = probas.size(1)
    losses = []
    class_to_sum = list(range(C)) if classes in ['all', 'present'] else classes
    for c in class_to_sum:
        fg = (labels == c).float()  # foreground for class c
        if (classes == 'present' and fg.sum() == 0):
            continue
        if C == 1:
            if len(classes) > 1:
                raise ValueError('Sigmoid output possible only with 1 class')
            class_pred = probas[:, 0]
        else:
            class_pred = probas[:, c]
        errors = (Variable(fg) - class_pred).abs()
        errors_sorted, perm = torch.sort(errors, 0, descending=True)
        perm = perm.data
        fg_sorted = fg[perm]
        losses.append(torch.dot(errors_sorted, Variable(lovasz_grad(fg_sorted))))
    return mean(losses)


def flatten_probas(probas, labels, ignore=None):
    """
    Flattens predictions in the batch
    """
    if probas.dim() == 3:
        # assumes output of a sigmoid layer
        B, H, W = probas.size()
        probas = probas.view(B, 1, H, W)
    B, C, H, W = probas.size()
    probas = probas.permute(0, 2, 3, 1).contiguous().view(-1, C)  # B * H * W, C = P, C
    labels = labels.view(-1)
    if ignore is None:
        return probas, labels
    valid = (labels != ignore)
    vprobas = probas[valid.nonzero(as_tuple=False).squeeze()]
    vlabels = labels[valid]
    return vprobas, vlabels


def xloss(logits, labels, ignore=None):
    """
    Cross entropy loss
    """
    return F.cross_entropy(logits, Variable(labels), ignore_index=255)


# --------------------------- HELPER FUNCTIONS ---------------------------
def isnan(x):
    return x != x


def mean(l, ignore_nan=False, empty=0):
    """
    nanmean compatible with generators.
    """
    l = iter(l)
    if ignore_nan:
        l = ifilterfalse(isnan, l)
    try:
        n = 1
        acc = next(l)
    except StopIteration:
        if empty == 'raise':
            raise ValueError('Empty mean')
        return empty
    for n, v in enumerate(l, 2):
        acc += v
    if n == 1:
        return acc
    return acc / n


def LovaszHingeLoss(inputs, target, num_classes=2):
    n, c, h, w = inputs.size()
    nt, ht, wt = target.size()
    if h != ht and w != wt:
        inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)

    # temp_inputs = inputs.argmax(axis=1).squeeze(1).float().requires_grad_(True)
    temp_inputs = torch.softmax(inputs, 1)[:, 1:2, ...].squeeze(1).contiguous()
    temp_target = target  # .unsqueeze(1)

    BinaryLovas_loss = lovasz_hinge(temp_inputs, temp_target, per_image=False)
    return BinaryLovas_loss


def LovaszSoftmaxLoss(inputs, target, num_classes=2):
    n, c, h, w = inputs.size()
    nt, ht, wt = target.size()
    if h != ht and w != wt:
        inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)

    temp_input = torch.sigmoid(inputs)  # (1, 2, 256, 256)#官方demo里的操作
    # temp_input  =  torch.softmax(inputs,1) #(1, 2, 256, 256)
    temp_target = target  # .unsqueeze(1)   #(1, 256, 256)
    # print(temp_input.detach().numpy().shape,temp_target.detach().numpy().shape)
    # print(temp_inputs.shape,temp_target.shape)
    # temp_inputs = temp_inputs.argmax(axis=-1)

    BinaryLovas_loss = lovasz_softmax(temp_input, temp_target, classes='present', ignore=num_classes)  # , classes=[1]
    return BinaryLovas_loss
import torch.nn as nn

class LovaszSoftmaxLoss(nn.Module):
    def __init__(self, num_classes=2, ignore=None):
        super().__init__()
        self.num_classes = num_classes
        self.ignore = ignore

    def forward(self, inputs, target):
        n, c, h, w = inputs.size()
        nt, ht, wt = target.size()
        if h != ht or w != wt:
            inputs = F.interpolate(inputs, size=(ht, wt),
                                   mode="bilinear", align_corners=True)

        temp_input = torch.sigmoid(inputs)
        return lovasz_softmax(temp_input, target,
                              classes='present',
                              ignore=self.ignore)


class LovaszHingeLoss(nn.Module):
    def __init__(self, num_classes=2, ignore=None):
        super().__init__()
        self.num_classes = num_classes
        self.ignore = ignore

    def forward(self, inputs, target):
        n, c, h, w = inputs.size()
        nt, ht, wt = target.size()
        if h != ht or w != wt:
            inputs = F.interpolate(inputs, size=(ht, wt),
                                   mode="bilinear", align_corners=True)

        temp_inputs = torch.softmax(inputs, 1)[:, 1:2, ...].squeeze(1).contiguous()
        return lovasz_hinge(temp_inputs, target, per_image=False)


# loop
def gen_matrix(pred_mask, gt_mask, num_classes=2):
    """
    gt_mask(ndarray): shape -> (height, width)
    pred_mask(ndarray):shape -> (height, width)
    num_classes:
    """
    pred_mask = torch.softmax(pred_mask.transpose(1, 2).transpose(2, 3).contiguous(), dim=-1)
    pred_mask = pred_mask.argmax(axis=-1)

    mask = (gt_mask >= 0) & (gt_mask < num_classes)
    count = torch.bincount(num_classes * gt_mask[mask].int() + pred_mask[mask], minlength=num_classes ** 2)

    cf_mtx = count.reshape(num_classes, num_classes)  # .cpu().numpy()
    oa, precision, recall, f_score, iou, miou, mcc = matrix2index(cf_mtx)

    return cf_mtx, oa, precision, recall, f_score, iou, miou, mcc


# gen_matrix
def matrix2index(matrix, smooth=1e-5):  # _macro
    cf_mtx = matrix
    smooth = torch.tensor(smooth, dtype=torch.float32).to(cf_mtx.device)
    ts1 = torch.tensor(1, dtype=torch.float32).to(cf_mtx.device)

    accuracy = torch.div(torch.sum(torch.diagonal(cf_mtx)), torch.maximum(torch.sum(cf_mtx), ts1))
    precision = torch.div(torch.diagonal(cf_mtx), torch.maximum(cf_mtx.sum(0), ts1))
    recall = torch.div(torch.diagonal(cf_mtx), torch.maximum(cf_mtx.sum(1), ts1))
    f_score = torch.div(2 * precision * recall, (precision + recall + smooth))

    miou = torch.div(torch.diag(cf_mtx),
                     (torch.sum(cf_mtx, axis=1) + torch.sum(cf_mtx, axis=0) - torch.diag(cf_mtx) + smooth))
    mcc = torch.div((torch.prod(torch.diagonal(cf_mtx)) - torch.prod(torch.diagonal(torch.fliplr(cf_mtx))) + smooth),
                    (torch.sqrt(torch.prod(cf_mtx.sum(0), dtype=torch.float32) * torch.prod(cf_mtx.sum(1),
                                                                                            dtype=torch.float32)) + smooth))

    oa = accuracy.item()
    precision = precision[1].item()
    recall = recall[1].item()
    f_score = f_score[1].item()
    iou = miou[1].item()
    miou = torch.mean(miou).item()
    mcc = mcc.item()

    return oa, precision, recall, f_score, iou, miou, mcc
