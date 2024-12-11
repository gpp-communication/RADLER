import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms
from collections import OrderedDict
from data_tools.downstream import DownstreamDataset
from models.ssl_encoder import radar_transform
from networks.downstream.radar_object_detector import RadarObjectDetector


def rename_keys(checkpoint):
    state_dict = torch.load(checkpoint, map_location='cpu')['state_dict']
    state_dict_new = OrderedDict()
    for key in state_dict.keys():
        if key.startswith('module.'):
            state_dict_new[key[7:]] = state_dict[key]
    del state_dict
    return state_dict_new


checkpoint_no_sdm = '/Users/yluo/Downloads/checkpoint_0059.pth.tar'
checkpoint_sdm = '/Users/yluo/Downloads/checkpoint_0059_sdm.pth.tar'

detector_sdm = RadarObjectDetector(None, 'test', 3, True)
state_dict_sdm = rename_keys(checkpoint_sdm)
detector_sdm.load_state_dict(state_dict_sdm)
detector_sdm.eval()

detector_no_sdm = RadarObjectDetector(None, 'test', 3)
state_dict_no_sdm = rename_keys(checkpoint_no_sdm)
detector_no_sdm.load_state_dict(state_dict_no_sdm)
detector_no_sdm.eval()

semantic_depth_transforms = transforms.Compose([transforms.ToTensor()])
radar_trans = radar_transform()
downstream_dataset = DownstreamDataset('../../../datasets/test/visualize_sdm/', radar_trans, semantic_depth_transforms)

with torch.no_grad():
    dataloader = DataLoader(downstream_dataset, batch_size=1, shuffle=True, num_workers=0)
    for image_path, radar, semantic_depth, gt_conf in dataloader:
        output_sdm = detector_sdm(radar, semantic_depth)
        output_no_sdm = detector_no_sdm(radar)
        # np.save(image_path[0].replace('IMAGES_0/', '').replace('.png', '_sdm.npy'), output_sdm)
        # np.save(image_path[0].replace('IMAGES_0/', '').replace('.png', '_no_sdm.npy'), output_no_sdm)
        diff = output_sdm - output_no_sdm
        diff = np.transpose(np.squeeze(diff), (1, 2, 0))
        plt.axis('off')
        plt.imshow(diff, vmin=0, vmax=1, origin='lower', aspect='auto')
        plt.savefig(image_path[0].replace('.png', '_diff.png').replace('IMAGES_0/', ''), bbox_inches='tight')
        plt.cla()

