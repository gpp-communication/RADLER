import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def visualize_training(fig_name, img_path, radar_data, output_confmap, gt_confmap):
    fig = plt.figure(figsize=(12, 12))
    img_data = Image.open(img_path)

    fig.add_subplot(2, 2, 1)
    plt.imshow(img_data)

    fig.add_subplot(2, 2, 2)
    plt.imshow(radar_data, origin='lower', aspect='auto')

    fig.add_subplot(2, 2, 3)
    output_confmap = np.transpose(output_confmap, (1, 2, 0))
    output_confmap[output_confmap < 0] = 0
    plt.imshow(output_confmap, vmin=0, vmax=1, origin='lower', aspect='auto')

    fig.add_subplot(2, 2, 4)
    confmap_gt = np.transpose(gt_confmap, (1, 2, 0))
    plt.imshow(confmap_gt, vmin=0, vmax=1, origin='lower', aspect='auto')

    plt.savefig(fig_name)
    plt.close(fig)


if __name__ == '__main__':
    fig_name = 'test'
    img_path = '/Users/yluo/Project/Radio-Vision-CityGML/datasets/test/test/IMAGES_0/000000.png'
    radar_path = '/Users/yluo/Project/Radio-Vision-CityGML/datasets/test/test/RADAR_RA_H/000000.npy'
    gt_confmap_path = '/Users/yluo/Project/Radio-Vision-CityGML/datasets/test/test/GT_CONFMAPS/000000.npy'
    radar_data = np.load(radar_path)
    output_confmap = torch.rand(3, 224, 221)
    gt_confmap = np.load(gt_confmap_path)
    visualize_training(fig_name, img_path, radar_data, output_confmap, gt_confmap[:3, :, :,])
