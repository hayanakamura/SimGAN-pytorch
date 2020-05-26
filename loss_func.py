import torch
import torch.nn as nn
import torch.nn.functional as F


def self_reg_loss(n, pred, true):
    lambda = n
    return lambda * torch.abs(pred - true)


def local_adv_loss(pred, true):
    true = torch.reshape(true,(-1,2))
    preds = torch.reshape(preds,(-1,2))
    loss = F.binary_cross_entropy_with_logits(true,preds)
    return loss
