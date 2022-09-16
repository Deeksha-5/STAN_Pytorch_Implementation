import torch
import torch.nn as nn

def dice_coef(y_true, y_pred):
    smooth = 1.
    y_true_f = nn.Flatten()(y_true)
    y_pred_f = nn.Flatten()(y_pred)
    intersection = torch.sum(y_true_f * y_pred_f)
    union = torch.sum(y_true_f) + torch.sum(y_pred_f)
    return (2. * intersection + smooth) / (union + smooth)


def true_pos(y_true, y_pred):
    smooth = 1
    y_pred_pos = torch.round(torch.clip(y_pred, 0, 1))
    y_pos = torch.round(torch.clip(y_true, 0, 1))
    tp = (torch.sum(y_pos * y_pred_pos) + smooth) / (torch.sum(y_pos) + smooth) 
    return tp 


def true_neg(y_true, y_pred):
    smooth = 1
    y_pred_pos = torch.round(torch.clip(y_pred, 0, 1))
    y_pred_neg = 1 - y_pred_pos
    y_pos = torch.round(torch.clip(y_true, 0, 1))
    y_neg = 1 - y_pos
    tn = torch.sum(y_neg * y_pred_neg)
    tn_ratio = (tn + smooth) / (K.sum(y_neg) + smooth)
    return tn_ratio


def false_pos(y_true, y_pred):
    smooth = 1
    y_pred_pos = torch.round(torch.clip(y_pred, 0, 1))
    y_pos = torch.round(torch.clip(y_true, 0, 1))
    y_neg = 1 - y_pos
    fp = torch.sum(y_neg * y_pred_pos)
    fp_ratio = (fp + smooth) / (torch.sum(y_neg) + smooth)
    return fp_ratio