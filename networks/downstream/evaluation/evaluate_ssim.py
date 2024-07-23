import os
import torch
import numpy as np
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM


def cal_ssim_and_ms_ssim(pred_path, gt_path):
    pred = np.load(pred_path)
    gt = np.load(gt_path)
    gt = gt[:3, :, :]
    pred = np.expand_dims(pred, 0)
    gt = np.expand_dims(gt, 0)
    pred = torch.from_numpy(pred)
    gt = torch.from_numpy(gt)
    # calculate ssim & ms-ssim for each image
    ssim_val = ssim(pred, gt, data_range=255, size_average=False)  # return (N,)
    ms_ssim_val = ms_ssim(pred, gt, data_range=255, size_average=False)  # (N,)

    return ssim_val, ms_ssim_val


def evaluate_ssim_and_ms_ssim(pred_folder, gt_folder):
    preds = [pred for pred in os.listdir(pred_folder) if pred.endswith('.npy')]
    preds = sorted(preds)
    gts = [gt for gt in os.listdir(gt_folder) if gt.endswith('.npy')]
    gts = sorted(gts)
    assert len(preds) == len(gts), 'predictions {} != ground truth {}'.format(len(preds), len(gts))
    ssim_sum, ms_sum = 0, 0
    for pred, gt in zip(preds, gts):
        ssim_tmp, ms_tmp = cal_ssim_and_ms_ssim(os.path.join(pred_folder, pred), os.path.join(gt_folder, gt))
        ssim_sum += ssim_tmp.numpy()[0]
        ms_sum += ms_tmp.numpy()[0]
    return ssim_sum / len(preds), ms_sum / len(preds)


if __name__ == '__main__':
    res_with_sd = ''
    res_without_sd = ''
    res_rodnet = '/Users/yluo/Downloads/rodnet-cdc-win16-tum-20240716-230710'
    ground_truth = '/Users/yluo/Pictures/CRTUM_new/data_cluster_1_2/downstream/test'
    sites = ['Arcisstrasse1', 'Arcisstrasse2', 'Arcisstrasse3', 'Arcisstrasse4',
             'Arcisstrasse5', 'Gabelsbergerstrasse1', 'Gabelsbergerstrasse2']
    evaluation_res = {'with_sd': {}, 'without_sd': {}, 'rodnet': {}}
    for site in sites:
        evaluation_res['with_sd'][site] = evaluate_ssim_and_ms_ssim(os.path.join(res_with_sd, site),
                                                                    os.path.join(ground_truth, site, 'GT_CONFMAPS'))
        evaluation_res['without_sd'][site] = evaluate_ssim_and_ms_ssim(os.path.join(res_without_sd, site),
                                                                    os.path.join(ground_truth, site, 'GT_CONFMAPS'))
        evaluation_res['rodnet'][site] = evaluate_ssim_and_ms_ssim(os.path.join(res_rodnet, site, 'rod_viz'),
                                                                    os.path.join(ground_truth, site, 'GT_CONFMAPS'))
    print(evaluation_res)
