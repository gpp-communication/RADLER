import math
import json
import numpy as np
import matplotlib.pyplot as plt

from .generate_grids import confmap2ra, ra2idx


def init_radar_json(n_frames):
    meta_all = []
    for frame_id in range(n_frames):
        meta_dict = dict(frame_id=frame_id)
        meta_dict['rad_h'] = dict(
                frame_name=None,
                n_objects=0,
                obj_info=dict(
                    anno_source=None,
                    categories=[],
                    centers=[],
                    center_ids=[],
                    scores=[]
                )
            )
    return meta_all


def get_class_id(class_str, classes):
    if class_str in classes:
        class_id = classes.index(class_str)
    else:
        if class_str == '':
            raise ValueError("No class name found")
        else:
            class_id = -1000
    return class_id


def load_anno_txt(txt_path, n_frame, dataset):
    anno_dict = init_radar_json(n_frame)
    with open(txt_path, 'r') as f:
        data = f.readlines()
    for line in data:
        frame_id, r, a, class_name = line.rstrip().split()
        frame_id = int(frame_id)
        r = float(r)
        a = float(a)
        rid, aid = ra2idx(r, a, dataset.range_grid, dataset.angle_grid)
        anno_dict[frame_id]['rad_h']['n_objects'] += 1
        anno_dict[frame_id]['rad_h']['obj_info']['categories'].append(class_name)
        anno_dict[frame_id]['rad_h']['obj_info']['centers'].append([r, a])
        anno_dict[frame_id]['rad_h']['obj_info']['center_ids'].append([rid, aid])
        anno_dict[frame_id]['rad_h']['obj_info']['scores'].append(1.0)

    return anno_dict


def normalize_confmap(confmap):
    conf_min = np.min(confmap)
    conf_max = np.max(confmap)
    if conf_max - conf_min != 0:
        confmap_norm = (confmap - conf_min) / (conf_max - conf_min)
    else:
        confmap_norm = confmap
    return confmap_norm


def add_noise_channel(confmap, radar_configs):
    n_class = 3

    confmap_new = np.zeros((n_class + 1, radar_configs['ramap_rsize'], radar_configs['ramap_asize']), dtype=float)
    confmap_new[:n_class, :, :] = confmap
    conf_max = np.max(confmap, axis=0)
    confmap_new[n_class, :, :] = 1.0 - conf_max
    return confmap_new


def visualize_confmap(confmap, pps=None):
    if pps is None:
        pps = []
    if len(confmap.shape) == 2:
        plt.imshow(confmap, origin='lower', aspect='auto')
        for pp in pps:
            plt.scatter(pp[1], pp[0], s=5, c='white')
        plt.show()
        return
    else:
        n_channel, _, _ = confmap.shape
    if n_channel == 3:
        confmap_viz = np.transpose(confmap, (1, 2, 0))
    elif n_channel > 3:
        confmap_viz = np.transpose(confmap[:3, :, :], (1, 2, 0))
        if n_channel == 4:
            confmap_noise = confmap[3, :, :]
            plt.imshow(confmap_noise, origin='lower', aspect='auto')
            plt.show()
    else:
        print("Warning: wrong shape of confmap!")
        return
    plt.imshow(confmap_viz, origin='lower', aspect='auto')
    for pp in pps:
        plt.scatter(pp[1], pp[0], s=5, c='white')
    plt.show()


def generate_confmaps(metadata_dict, n_class, viz):
    with open('../configs/radar_config.json') as f:
        radar_configs = json.load(f)
    with open('../configs/object_config.json') as f:
        object_config = json.load(f)
    confmaps = []
    for metadata_frame in metadata_dict:
        n_obj = metadata_frame['rad_h']['n_objects']
        obj_info = metadata_frame['rad_h']['obj_info']
        if n_obj == 0:
            confmap_gt = np.zeros(
                (n_class + 1, radar_configs['ramap_rsize'], radar_configs['ramap_asize']),
                dtype=float)
            confmap_gt[-1, :, :] = 1.0  # initialize noise channal
        else:
            confmap_gt = generate_confmap(n_obj, obj_info, object_config)
            confmap_gt = normalize_confmap(confmap_gt)
            confmap_gt = add_noise_channel(confmap_gt, radar_configs)
        assert confmap_gt.shape == (
            n_class + 1, radar_configs['ramap_rsize'], radar_configs['ramap_asize'])
        if viz:
            visualize_confmap(confmap_gt)
        confmaps.append(confmap_gt)
    confmaps = np.array(confmaps)
    return confmaps


def generate_confmap(n_obj, obj_info, config_dict, gaussian_thres=36):
    n_class = 3
    classes = ['pedestrian', 'car', 'cyclist']  # TODO: check the order of the object classes in the array
    ramap_rsize = 224
    ramap_asize = 221
    confmap_sigmas = config_dict['confmap_cfg']['confmap_sigmas']
    confmap_sigmas_interval = config_dict['confmap_cfg']['confmap_sigmas_interval']
    confmap_length = config_dict['confmap_cfg']['confmap_length']

    range_grid = confmap2ra('range')

    confmap = np.zeros((n_class, ramap_rsize, ramap_asize), dtype=float)
    for objid in range(n_obj):
        rng_idx = obj_info['center_ids'][objid][0]
        agl_idx = obj_info['center_ids'][objid][1]
        class_name = obj_info['categories'][objid]
        if class_name not in classes:
            # print("not recognized class: %s" % class_name)
            continue
        class_id = get_class_id(class_name, classes)
        sigma = 2 * np.arctan(confmap_length[class_name] / (2 * range_grid[rng_idx])) * confmap_sigmas[class_name]
        sigma_interval = confmap_sigmas_interval[class_name]
        if sigma > sigma_interval[1]:
            sigma = sigma_interval[1]
        if sigma < sigma_interval[0]:
            sigma = sigma_interval[0]
        for i in range(ramap_rsize):
            for j in range(ramap_asize):
                distant = (((rng_idx - i) * 2) ** 2 + (agl_idx - j) ** 2) / sigma ** 2
                if distant < gaussian_thres:  # threshold for confidence maps
                    value = np.exp(- distant / 2) / (2 * math.pi)
                    confmap[class_id, i, j] = value if value > confmap[class_id, i, j] else confmap[class_id, i, j]

    return confmap


if __name__ == '__main__':
    pass
