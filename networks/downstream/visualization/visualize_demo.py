import os
import json
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

from networks.downstream.post_processing import get_class_name, post_process_single_frame


def visualize_demo_img(fig_name, img_path, confmap_no_sdm, res_final_no_sdm, confmap_sdm, res_final_sdm):
    with open('/Users/yluo/Project/Radio-Vision-CityGML/networks/downstream/configs/object_config.json', 'r') as f:
        object_cfg = json.load(f)
    classes = object_cfg['classes']

    fig = plt.figure(figsize=(20, 8))
    # Use GridSpec to control the layout of the subplots
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(1, 3, width_ratios=[1.25, 1, 1])  # The first subplot will be 2x larger than the others
    img_data = Image.open(img_path)

    fig.add_subplot(gs[0])
    plt.imshow(img_data)
    plt.axis('off')
    plt.title("Image")

    # without sdm
    fig.add_subplot(gs[1])
    max_dets_without, _ = res_final_no_sdm.shape
    output_confmap = np.transpose(confmap_no_sdm, (1, 2, 0))
    output_confmap[output_confmap < 0] = 0
    # output_confmap[:] = 1
    plt.imshow(output_confmap, vmin=0, vmax=1, origin='lower', aspect='auto')
    colors = ['red', 'green', 'blue']
    for d in range(max_dets_without):
        cla_id = int(res_final_no_sdm[d, 0])
        if cla_id == -1:
            continue
        row_id = res_final_no_sdm[d, 1]
        col_id = res_final_no_sdm[d, 2]
        conf = res_final_no_sdm[d, 3]
        conf = 1.0 if conf > 1 else conf
        cla_str = get_class_name(cla_id, classes)
        plt.scatter(col_id, row_id, s=10, c='white')
        text = cla_str + '\n%.2f' % conf
        plt.text(col_id + 5, row_id, text, color='white', fontsize=10)
    plt.grid(color='#0f0f0f', linestyle='--', linewidth=0.8)
    plt.axis('off')
    plt.title("Without SDM")

    fig.add_subplot(gs[2])
    max_dets, _ = res_final_sdm.shape
    output_confmap = np.transpose(confmap_sdm, (1, 2, 0))
    output_confmap[output_confmap < 0] = 0
    # output_confmap[:] = 1
    plt.imshow(output_confmap, vmin=0, vmax=1, origin='lower', aspect='auto')
    colors = ['red', 'green', 'blue']
    for d in range(max_dets):
        cla_id = int(res_final_sdm[d, 0])
        if cla_id == -1:
            continue
        row_id = res_final_sdm[d, 1]
        col_id = res_final_sdm[d, 2]
        conf = res_final_sdm[d, 3]
        conf = 1.0 if conf > 1 else conf
        cla_str = get_class_name(cla_id, classes)
        plt.scatter(col_id, row_id, s=10, c='white')
        text = cla_str + '\n%.2f' % conf
        plt.text(col_id + 5, row_id, text, color='white', fontsize=10)
    plt.grid(color='#0f0f0f', linestyle='--', linewidth=0.8)
    plt.axis('off')
    plt.title("With SDM")

    # Use tight_layout with custom padding to reduce white space
    plt.tight_layout(pad=8, h_pad=2, w_pad=2)

    plt.savefig(fig_name)
    plt.close(fig)


if __name__ == '__main__':
    sites = ['Arcisstrasse1', 'Arcisstrasse2', 'Arcisstrasse3', 'Arcisstrasse4',
             'Arcisstrasse5', 'Gabelsbergerstrasse1', 'Gabelsbergerstrasse2']
    base_folder = '/Users/yluo/Pictures/CRTUM/test'
    for site in sites:
        images = [image for image in os.listdir(os.path.join(base_folder, site, 'IMAGES_0')) if image.endswith('.png')]
        for image in tqdm(images, desc=site):
            img_path = os.path.join(base_folder, site, 'IMAGES_0', image)
            confmap_no_sdm_path = os.path.join(base_folder, 'no_sdm', site, image.replace('.png', '.npy'))
            confmap_sdm_path = os.path.join(base_folder, 'sdm', site, image.replace('.png', '.npy'))
            confmap_no_sdm = np.squeeze(np.load(confmap_no_sdm_path))
            confmap_sdm = np.squeeze(np.load(confmap_sdm_path))
            results_no_sdm = post_process_single_frame(confmap_no_sdm)
            results_sdm = post_process_single_frame(confmap_sdm)
            fig_name = os.path.join(base_folder, site, image)
            visualize_demo_img(fig_name, img_path, confmap_no_sdm, results_no_sdm, confmap_sdm, results_sdm)
