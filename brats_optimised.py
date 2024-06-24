import numpy as np
import cc3d

total_time = []

def dice(im1, im2):
    intersection = np.sum(im1 * im2)
    sum_im1 = np.sum(im1)
    sum_im2 = np.sum(im2)
    return 2.0 * intersection / (sum_im1 + sum_im2)

def lesion_dice_gpu(pred, gt):

    pred = pred.cpu().numpy()
    gt = gt.cpu().numpy()    

    pred_label_cc = cc3d.connected_components(pred, connectivity=26)
    gt_label_cc = cc3d.connected_components(gt, connectivity=26)

    num_gt_lesions = len(np.unique(gt_label_cc[gt_label_cc != 0]))

    lesion_dice_scores = 0
    tp = np.array([], dtype=int)
    fn = np.array([], dtype=int)

    for gtcomp in range(1, num_gt_lesions + 1):
        gt_tmp = (gt_label_cc == gtcomp)
        intersecting_cc = np.unique(pred_label_cc[gt_tmp])
        intersecting_cc = intersecting_cc[intersecting_cc != 0]

        if len(intersecting_cc) > 0:
            pred_tmp = np.zeros_like(pred_label_cc, dtype=bool)
            pred_tmp[np.isin(pred_label_cc, intersecting_cc)] = True
            dice_score = dice(pred_tmp, gt_tmp)
            lesion_dice_scores += dice_score
            tp = np.concatenate([tp, intersecting_cc])
        else:
            fn = np.append(fn, gtcomp)
    
    fp = np.unique(pred_label_cc[np.isin(pred_label_cc,tp+[0],invert=True)])
    fp = fp[fp != 0]

    return lesion_dice_scores / (num_gt_lesions + len(fp))