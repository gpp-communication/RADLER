import torch
import numpy as np
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM

rodnet = np.load('/Users/yluo/Downloads/rodnet-cdc-win16-tum-20240716-230710/Arcisstrasse1/rod_viz/000000.npy')
gt = np.load('/Users/yluo/Pictures/CRTUM_new/data_cluster_1_2/downstream/test/Arcisstrasse1/GT_CONFMAPS/000000.npy')
gt = gt[:3, :, :]
rodnet = np.expand_dims(rodnet, 0)
gt = np.expand_dims(gt, 0)
rodnet = torch.from_numpy(rodnet)
gt = torch.from_numpy(gt)
# calculate ssim & ms-ssim for each image
ssim_val = ssim( rodnet, gt, data_range=255, size_average=False) # return (N,)
ms_ssim_val = ms_ssim( rodnet, gt, data_range=255, size_average=False ) #(N,)

print('SSIM: ', ssim_val)
print('MS-SSIM: ', ms_ssim_val)