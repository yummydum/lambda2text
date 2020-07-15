from pathlib import Path
import json
from IPython.core.display import display, HTML, Javascript
import os
import pandas as pd
import torch
import torch.nn as nn
from transformers.optimization import AdamW, get_linear_schedule_with_warmup

DATA_DIR = Path(__file__).absolute().parent.parent / 'data/ETPC'


def get_optimzer(model, lr, decay=0.0):
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay":
            decay,
        },
        {
            "params": [
                p for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay":
            0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=lr)
    return optimizer


def get_scheduler(optimizer, train_step, warmup_step):
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=warmup_step,
                                                num_training_steps=train_step)
    return scheduler


def cross_entropy(pred, target):

    # Copy to avoid destructive operation
    pred = pred.clone()
    target = target.clone()

    # Check shape match
    msg = f'Illegal shape pred:{pred.size()}, {target.size()}'
    assert pred.size() == target.size(), msg

    if target.type() in {'torch.LongTensor', 'torch.cuda.LongTensor'}:
        target = target.float()

    m = nn.Sigmoid()
    loss = nn.BCELoss(reduction='sum')
    return loss(m(pred), target)


def hinge_loss(pred, target):
    """
    Element-wise hinge loss
    """

    # Copy to avoid destructive operation
    pred = pred.clone()
    target = target.clone()

    # Check shape match
    msg = f'Illegal shape pred:{pred.size()}, {target.size()}'
    assert pred.size() == target.size(), msg

    # 0 -> -1 for hinge loss
    target[target == 0] = -1

    # Convert Long labels to float
    if target.type() in {'torch.LongTensor', 'torch.cuda.LongTensor'}:
        target = target.float()

    # Calc loss
    loss = 1 - (pred * target)
    loss[loss < 0] = 0
    return torch.sum(loss)


def get_pred(pred, threshold=0):
    pred = pred.clone().detach()
    pred[pred > threshold] = 1
    pred[pred <= threshold] = 0
    pred = pred.long()
    return pred


def calc_acc(pred, target):
    msg = f'Illegal shape pred:{pred.size()}, {target.size()}'
    assert pred.size() == target.size(), msg
    return torch.sum(pred == target, dim=0)


def calc_recall(pred, target):
    msg = f'Illegal shape pred:{pred.size()}, {target.size()}'
    assert pred.size() == target.size(), msg
    result = torch.zeros(pred.size(1), dtype=torch.long)
    if pred.is_cuda:
        result = result.to(pred.get_device())
    for i in range(pred.size(1)):
        b = target[:, i] == 1
        result[i] = pred[:, i][b].sum()
    return result
