import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from networks.downstream.post_processing import get_class_name, post_process_single_frame


def visualize_training(fig_name, img_path, radar_data, output_confmap, gt_confmap):
    fig = plt.figure(figsize=(12, 12))
    img_data = Image.open(img_path)

    fig.add_subplot(2, 2, 1)
    plt.imshow(img_data)

    fig.add_subplot(2, 2, 2)
    radar_data[radar_data == 0] = 1
    data = np.log10(radar_data) * 20
    max_value = np.max(data)
    im = plt.imshow(data, origin='lower', aspect='auto', cmap='jet')
    im.set_clim(max_value - 30, max_value)

    fig.add_subplot(2, 2, 3)
    output_confmap = np.transpose(output_confmap, (1, 2, 0))
    output_confmap[output_confmap < 0] = 0
    plt.imshow(output_confmap, vmin=0, vmax=1, origin='lower', aspect='auto')

    fig.add_subplot(2, 2, 4)
    confmap_gt = np.transpose(gt_confmap, (1, 2, 0))
    plt.imshow(confmap_gt, vmin=0, vmax=1, origin='lower', aspect='auto')

    plt.savefig(fig_name)
    plt.close(fig)


def visualize_test_img(fig_name, img_path, radar_data, output_confmap, gt_confmap, res_final):
    max_dets, _ = res_final.shape
    with open('/home/stud/luoyu/storage/user/luoyu/projects/Radio-Vision-CityGML/networks/downstream/configs/object_config.json', 'r') as f:
        object_cfg = json.load(f)
    classes = object_cfg['classes']

    fig = plt.figure(figsize=(12, 12))
    img_data = Image.open(img_path)

    fig.add_subplot(2, 2, 1)
    plt.imshow(img_data)
    plt.axis('off')
    plt.title("Image")

    fig.add_subplot(2, 2, 2)
    radar_data[radar_data == 0] = 1
    data = np.log10(radar_data) * 20
    max_value = np.max(data)
    im = plt.imshow(data, origin='lower', aspect='auto', cmap='jet')
    im.set_clim(max_value - 30, max_value)
    plt.axis('off')
    plt.title("RA Heatmap")

    fig.add_subplot(2, 2, 3)
    output_confmap = np.transpose(output_confmap, (1, 2, 0))
    output_confmap[output_confmap < 0] = 0
    plt.imshow(output_confmap, vmin=0, vmax=1, origin='lower', aspect='auto')
    for d in range(max_dets):
        cla_id = int(res_final[d, 0])
        if cla_id == -1:
            continue
        row_id = res_final[d, 1]
        col_id = res_final[d, 2]
        conf = res_final[d, 3]
        conf = 1.0 if conf > 1 else conf
        cla_str = get_class_name(cla_id, classes)
        plt.scatter(col_id, row_id, s=10, c='white')
        text = cla_str + '\n%.2f' % conf
        plt.text(col_id + 5, row_id, text, color='white', fontsize=10)
    plt.axis('off')
    plt.title("Downstream Detection")

    fig.add_subplot(2, 2, 4)
    confmap_gt = np.transpose(gt_confmap, (1, 2, 0))
    plt.imshow(confmap_gt, vmin=0, vmax=1, origin='lower', aspect='auto')
    plt.axis('off')
    plt.title("Ground Truth")

    plt.savefig(fig_name)
    plt.close(fig)


if __name__ == '__main__':
    fig_name = 'test1'
    img_path = '/datasets/test/test-1/IMAGES_0/000000.png'
    radar_path = '/datasets/test/test-1/RADAR_RA_H/000000.npy'
    gt_confmap_path = '/datasets/test/test-1/GT_CONFMAPS/000000.npy'
    radar_data = np.load(radar_path)
    output_confmap = torch.rand(3, 224, 221)
    results = post_process_single_frame(output_confmap)
    gt_confmap = np.load(gt_confmap_path)
    visualize_test_img(fig_name, img_path, radar_data, output_confmap, gt_confmap[:3, :, :,], results)
