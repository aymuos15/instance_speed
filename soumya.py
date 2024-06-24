import torch
import numpy as np

import cc3d

total_time = []

def dice(im1, im2):
    intersection = torch.sum(im1 * im2)
    sum_im1 = torch.sum(im1)
    sum_im2 = torch.sum(im2)
    return 2.0 * intersection / (sum_im1 + sum_im2)

def lesion_dice_gpu(pred, gt):

    pred_label_cc = cc3d.connected_components(pred.cpu().numpy(), connectivity=26)
    gt_label_cc = cc3d.connected_components(gt.cpu().numpy(), connectivity=26)

    pred_label_cc = torch.tensor(pred_label_cc.astype(np.float64))
    gt_label_cc = torch.tensor(gt_label_cc.astype(np.float64))

    num_gt_lesions = torch.unique(gt_label_cc[gt_label_cc != 0]).size(0)

    lesion_dice_scores = 0
    tp = torch.tensor([])
    fn = torch.tensor([])

    for gtcomp in range(1, num_gt_lesions + 1):
        gt_tmp = (gt_label_cc == gtcomp)
        intersecting_cc = torch.unique(pred_label_cc[gt_tmp])
        intersecting_cc = intersecting_cc[intersecting_cc != 0]

        if len(intersecting_cc) > 0:
            pred_tmp = torch.zeros_like(pred_label_cc, dtype=torch.bool)
            pred_tmp[torch.isin(pred_label_cc, intersecting_cc)] = True
            dice_score = dice(pred_tmp, gt_tmp)
            lesion_dice_scores += dice_score
            tp = torch.cat([tp, intersecting_cc])
        else:
            fn = torch.cat([fn, torch.tensor([gtcomp])])
    
    mask = (pred_label_cc != 0) & (~torch.isin(pred_label_cc, tp))
    fp = torch.unique(pred_label_cc[mask], sorted=True)
    fp = fp[fp != 0]

    return lesion_dice_scores / (num_gt_lesions + len(fp))