import math
import json
import torch
import numpy as np

from networks.downstream.confidence_map import confmap2ra
from networks.downstream.radar_object_detector import RadarObjectDetector


def get_class_name(class_id, classes):
    n_class = len(classes)
    if 0 <= class_id < n_class:
        class_name = classes[class_id]
    elif class_id == -1000:
        class_name = '__background'
    else:
        raise ValueError("Class ID is not defined")
    return class_name


def pol2cart_ramap(rho, phi):
    """
    Transform from polar to cart under RAMap coordinates
    :param rho: distance to origin
    :param phi: angle (rad) under RAMap coordinates
    :return: x, y
    """
    x = rho * np.sin(phi)
    y = rho * np.cos(phi)
    return x, y


def detect_peaks(image, search_size, threshold=0.3):
    peaks_row = []
    peaks_col = []
    height, width = image.shape
    for h in range(search_size - 1, height - (search_size - 1)):
        for w in range(search_size, width - search_size):
            # this is to make sure "center" is the only one peak in this area
            area = image[h - search_size + 1:h + search_size, w - search_size:w + search_size + 1]
            center = image[h, w]
            flag = np.where(area >= center)
            if flag[0].shape[0] == 1 and center > threshold:
                peaks_row.append(h)
                peaks_col.append(w)

    return peaks_row, peaks_col


def get_ols_btw_objects(obj1, obj2):
    with open('../../../networks/downstream/configs/object_config.json') as f:
        object_cfg = json.load(f)

    classes = object_cfg["classes"]
    object_sizes = object_cfg["sizes"]

    if obj1['class_id'] != obj2['class_id']:
        print('Error: Computing OLS between different classes!')
        raise TypeError("OLS can only be compute between objects with same class.  ")
    if obj1['score'] < obj2['score']:
        raise TypeError("Confidence score of obj1 should not be smaller than obj2. "
                        "obj1['score'] = %s, obj2['score'] = %s" % (obj1['score'], obj2['score']))

    classid = obj1['class_id']
    class_str = get_class_name(classid, classes)
    rng1 = obj1['range']
    agl1 = obj1['angle']
    rng2 = obj2['range']
    agl2 = obj2['angle']
    x1, y1 = pol2cart_ramap(rng1, agl1)
    x2, y2 = pol2cart_ramap(rng2, agl2)
    dx = x1 - x2
    dy = y1 - y2
    s_square = x1 ** 2 + y1 ** 2
    kappa = object_sizes[class_str] / 100  # TODO: tune kappa
    e = (dx ** 2 + dy ** 2) / 2 / (s_square * kappa)
    ols = math.exp(-e)
    return ols


def lnms(obj_dicts_in_class):
    """
    Location-based NMS
    :param obj_dicts_in_class:
    :param config_dict:
    :return:
    """
    with open('../../../networks/downstream/configs/model_config.json') as f:
        model_configs = json.load(f)
    detect_mat = - np.ones((model_configs['max_dets'], 4))
    cur_det_id = 0
    # sort peaks by confidence score
    inds = np.argsort([-d['score'] for d in obj_dicts_in_class], kind='mergesort')
    dts = [obj_dicts_in_class[i] for i in inds]
    while len(dts) != 0:
        if cur_det_id >= model_configs['max_dets']:
            break
        p_star = dts[0]
        detect_mat[cur_det_id, 0] = p_star['class_id']
        detect_mat[cur_det_id, 1] = p_star['range_id']
        detect_mat[cur_det_id, 2] = p_star['angle_id']
        detect_mat[cur_det_id, 3] = p_star['score']
        cur_det_id += 1
        del dts[0]
        for pid, pi in enumerate(dts):
            ols = get_ols_btw_objects(p_star, pi)
            # dts are sorted by the confidence score, so dts[pid] has smaller confidence score --> can be deleted
            if ols > model_configs['ols_thres']:
                del dts[pid]

    return detect_mat


def post_process_single_frame(confmap):
    """
    Post-processing for RODNet
    :param confmap: predicted confidence map [n_class, ramap_r, ramap_a]
    :return: [1, max_dets, 4]
    """
    with open('../../../networks/downstream/configs/radar_config.json') as f:
        radar_configs = json.load(f)
    n_class = 3
    rng_grid = confmap2ra('range', radar_configs)
    agl_grid = confmap2ra('angle', radar_configs)
    with open('../../../networks/downstream/configs/model_config.json') as f:
        model_configs = json.load(f)
    max_dets = model_configs['max_dets']
    peak_thres = model_configs['peak_thres']
    search_size = 2

    class_size, height, width = confmap.shape

    if class_size != n_class:
        raise TypeError("Wrong class number setting. ")

    res_final = - np.ones((max_dets, 4))

    detect_mat = []
    for c in range(class_size):
        obj_dicts_in_class = []
        confmap_class = confmap[c, :, :]
        rowids, colids = detect_peaks(confmap_class, search_size, peak_thres)

        for ridx, aidx in zip(rowids, colids):
            rng = rng_grid[ridx]
            agl = agl_grid[aidx]
            conf = confmap_class[ridx, aidx]
            obj_dict = dict(
                frame_id=None,
                range=rng,
                angle=agl,
                range_id=ridx,
                angle_id=aidx,
                class_id=c,
                score=conf,
            )
            obj_dicts_in_class.append(obj_dict)

        detect_mat_in_class = lnms(obj_dicts_in_class)
        detect_mat.append(detect_mat_in_class)

    detect_mat = np.array(detect_mat)
    detect_mat = np.reshape(detect_mat, (class_size * max_dets, 4))
    detect_mat = detect_mat[detect_mat[:, 3].argsort(kind='mergesort')[::-1]]
    res_final[:, :] = detect_mat[:max_dets]

    return res_final


if __name__ == '__main__':
    output = np.load('../../../datasets/test/visualize_sdm/test-1/000095_sdm.npy')
    for i in range(output.shape[0]):
        results = post_process_single_frame(output[i])
        print(i, results)
