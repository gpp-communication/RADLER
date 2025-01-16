import os
import json
import math
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

from networks.downstream.post_processing import get_class_name, post_process_single_frame


def visualize_demo_img(fig_name, img_path, confmap_no_sdm, res_final_no_sdm, confmap_sdm, res_final_sdm):
    with open('../../../networks/downstream/configs/object_config.json', 'r') as f:
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
        color = 'white'
        fontsize = 12
        if res_final_no_sdm[d, -1] == 'Array 1':
            color = 'red'
            fontsize = 14
        if round(conf, 2) == 1.00:
            color = 'white'
            fontsize = 12
        plt.text(col_id + 5, row_id, text, color=color, fontsize=fontsize)
    plt.grid(color='#0f0f0f', linestyle='--', linewidth=0.8)
    plt.axis('off')
    plt.title("Without SDM")

    fig.add_subplot(gs[2])
    max_dets, _ = res_final_sdm.shape
    output_confmap = np.transpose(confmap_sdm, (1, 2, 0))
    output_confmap[output_confmap < 0] = 0
    # output_confmap[:] = 1
    plt.imshow(output_confmap, vmin=0, vmax=1, origin='lower', aspect='auto')
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
        color = 'white'
        fontsize = 12
        if res_final_sdm[d, -1] == 'Array 2':
            color = 'red'
            fontsize = 14
        if round(conf, 2) == 1.00:
            color = 'white'
            fontsize = 12
        plt.text(col_id + 5, row_id, text, color=color, fontsize=fontsize)
    plt.grid(color='#0f0f0f', linestyle='--', linewidth=0.8)
    plt.axis('off')
    plt.title("With SDM")

    # Use tight_layout with custom padding to reduce white space
    plt.tight_layout(pad=8, h_pad=2, w_pad=2)

    plt.savefig(fig_name)
    plt.close(fig)


# Define a function to calculate the Euclidean distance between two points
def euclidean_distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


# Define a function to compare the two arrays and attach the higher confidence array info
def compare_and_attach(array1, array2, threshold=5):
    # Create copies of both arrays to avoid modifying them during comparison
    array1_copy = np.array(array1, dtype=object)
    array2_copy = np.array(array2, dtype=object)
    array2_copy_2 = np.array(array2, dtype=object)
    indicators = [[''] for _ in range(array1.shape[0])]
    array1_copy = np.append(array1_copy, indicators, axis=1)
    array2_copy = np.append(array2_copy, indicators, axis=1)
    array2_copy_2 = np.append(array2_copy_2, indicators, axis=1)

    # Loop through each entry in the first array
    for i, obj1 in enumerate(array1_copy):
        class1, x1, y1, conf1, _ = obj1
        if class1 == -1.0:  # Skip entries with class -1.0
            continue
        # Find the closest match in the second array within the threshold
        for j, obj2 in enumerate(array2_copy):
            class2, x2, y2, conf2, _ = obj2
            if class2 == -1.0:  # Skip entries with class -1.0
                continue
            # Calculate the Euclidean distance between the two objects
            distance = euclidean_distance(x1, y1, x2, y2)

            # If the distance is less than the threshold, consider them a match
            if distance <= threshold and class1 == class2:
                # Compare confidence values and attach the higher confidence array info
                if round(conf1, 2) > round(conf2, 2):
                    array1_copy[i][-1] = 'Array 1'  # Attach info to array1
                    for obj in array2_copy_2:
                        if x2 == obj[1] and y2 == obj[2]:
                            obj[-1] = 'Array 1'
                else:
                    array1_copy[i][-1] = 'Array 2'  # Attach info to array1
                    for obj in array2_copy_2:
                        if x2 == obj[1] and y2 == obj[2]:
                            obj[-1] = 'Array 2'

                # Remove the matched object from array2_copy (avoid re-matching)
                array2_copy = np.delete(array2_copy, j, 0)
                break

    for obj2 in array2_copy_2:
        if obj2[-1] == '' and obj2[0] == -1.00:
            obj2[-1] = 'Array 2'

    return array1_copy, array2_copy_2


if __name__ == '__main__':
    sites = ['Arcisstrasse1', 'Arcisstrasse2', 'Arcisstrasse3', 'Arcisstrasse4',
             'Arcisstrasse5', 'Gabelsbergerstrasse1', 'Gabelsbergerstrasse2']
    base_folder = '/Users/yluo/Pictures/CRCTUM/test'
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
            results_no_sdm, results_sdm = compare_and_attach(results_no_sdm, results_sdm, 7)
            fig_name = os.path.join(base_folder, site, image)
            visualize_demo_img(fig_name, img_path, confmap_no_sdm, results_no_sdm, confmap_sdm, results_sdm)
