import os
import json
import numpy as np

from .evaluate_ols import read_gt_txt, read_sub_txt


def evaluate_localization_error(data_path, submit_dir, truth_dir):
    with open('../configs/object_config.json', 'r') as f:
        object_config = json.load(f)
    sub_names = sorted(os.listdir(submit_dir))
    gt_names = sorted(os.listdir(truth_dir))
    assert len(sub_names) == len(gt_names), "missing submission files!"
    for sub_name, gt_name in zip(sub_names, gt_names):
        if sub_name != gt_name:
            raise AssertionError("wrong submission file names!")

    for seqid, (sub_name, gt_name) in enumerate(zip(sub_names, gt_names)):
        gt_path = os.path.join(truth_dir, gt_name)
        sub_path = os.path.join(submit_dir, sub_name)
        n_frame = len(os.listdir(os.path.join(data_path, gt_name.rstrip('.txt'), 'IMAGES_0')))
        gt_dets = read_gt_txt(gt_path, n_frame, object_config)
        sub_dets = read_sub_txt(sub_path, n_frame, object_config)




if __name__ == '__main__':
    pass
