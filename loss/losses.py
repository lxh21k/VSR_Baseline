from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from common.utils import AverageMeter


class LossL1(nn.Module):
    def __init__(self):
        super(LossL1, self).__init__()
        self.loss = nn.L1Loss()

    def __call__(self, input, target):
        return self.loss(input, target)


class LossL2(nn.Module):
    def __init__(self):
        super(LossL2, self).__init__()
        self.loss = nn.MSELoss()

    def __call__(self, input, target):
        return self.loss(input, target)


class LossSmoothL1(nn.Module):
    def __init__(self):
        super(LossSmoothL1, self).__init__()
        self.loss = nn.SmoothL1Loss()

    def __call__(self, input, target):
        return self.loss(input, target)


class LossCrossEntropy(nn.Module):
    def __init__(self, weight=None):
        super(LossCrossEntropy, self).__init__()
        self.loss = torch.nn.CrossEntropyLoss(weight=weight)

    def __call__(self, input, target, weight=None):
        return self.loss(input, target)


def compute_losses(data, endpoints, params):
    loss = {}

    # compute losses
    if params.loss_type == "basic":
        ce_criterion = LossCrossEntropy()

        B = data["label"].size()[0]
        pred = endpoints["p"]
        label = data["label"]
        loss['total'] = ce_criterion(pred, label)
    elif params.loss_type == "pixel_loss":

        l1 = LossL1()

        gt = data['gt']
        pred = endpoints

        loss['total'] = l1(pred, gt)

    else:
        raise NotImplementedError
    return loss


def compute_metrics(data, endpoints, manager):
    metrics = {}
    with torch.no_grad():
        # compute metrics
        B = data["label"].size()[0]
        outputs = np.argmax(endpoints["p"].detach().cpu().numpy(), axis=1)
        accuracy = np.sum(outputs.astype(np.int32) == data["label"].detach().cpu().numpy().astype(np.int32)) / B
        metrics['accuracy'] = accuracy
        return metrics
