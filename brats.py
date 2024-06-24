import numpy as np
import cc3d

def dice(im1, im2):
    intersection = np.logical_and(im1, im2)
    return 2. * (intersection.sum()) / (im1.sum() + im2.sum())

def lesion_dice(pred, gt):

    pred_label_cc = cc3d.connected_components(pred.cpu().numpy(), connectivity=26)
    gt_label_cc = cc3d.connected_components(gt.cpu().numpy(), connectivity=26)

    fp = []
    tp = []
    fn = []
    lesion_dice_scores = []

    for gtcomp in range(1, np.max(gt_label_cc) + 1):

        gt_tmp = np.zeros_like(gt_label_cc)
        gt_tmp[gt_label_cc == gtcomp] = 1
        
        pred_tmp = np.copy(pred_label_cc)
        pred_tmp = pred_tmp*gt_tmp

        intersecting_cc = np.unique(pred_tmp) 
        intersecting_cc = intersecting_cc[intersecting_cc != 0] 

        for cc in intersecting_cc:
            tp.append(cc)

        pred_tmp = np.copy(pred_label_cc)
        pred_tmp[np.isin(pred_tmp,intersecting_cc,invert=True)] = 0
        pred_tmp[np.isin(pred_tmp,intersecting_cc)] = 1

        dice_score = dice(pred_tmp, gt_tmp)
        
        lesion_dice_scores.append(dice_score)
        
        if len(intersecting_cc) > 0:
            pass
        else:
            fn.append(gtcomp)

    fp = np.unique(pred_label_cc[np.isin(pred_label_cc,tp+[0],invert=True)])
    
    return sum(lesion_dice_scores) /((len(lesion_dice_scores)) + len(fp))