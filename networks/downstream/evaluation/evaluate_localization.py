import os
import json
import numpy as np

from networks.downstream.evaluation.evaluate_ols import read_gt_txt, read_sub_txt


def calculate_localization_error(sub_dets: dict, gt_dets:dict, num_frames: int):
    errors_all = [[], [], []]
    for i in range(num_frames):
        for c in range(3):  # pedestrian, cyclist, car
            dets = []
            gts = []
            if len(sub_dets[i, c]) > 0 and len(gt_dets[i, c]) > 0:
                for det in sub_dets[i, c]:
                    angle = np.rad2deg(det['angle'])
                    dets.append((det['range'] * np.cos(angle), det['range'] * np.sin(angle)))  # (x, y)

                for gt in gt_dets[i, c]:
                    angle = np.rad2deg(gt['angle'])
                    gts.append((gt['range'] * np.cos(angle), gt['range'] * np.sin(angle)))

                dets.sort(key=lambda x: x[0])
                gts.sort(key=lambda x: x[0])

                error_sum = 0
                for j in range(min(len(dets), len(gts))):
                    mse = np.sqrt(np.sum(np.power([dets[j][0] - gts[j][0], gts[j][1] - gts[j][1]], 2)))
                    error_sum += mse

                errors_all[c].append(error_sum / min(len(dets), len(gts)))
    print(errors_all[0])


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
        calculate_localization_error(sub_dets, gt_dets, n_frame)


if __name__ == '__main__':
    data_path = './test_localization/'
    submit_dir = './test_localization/prediction'
    truth_dir = './test_localization/gt'
    evaluate_localization_error(data_path, submit_dir, truth_dir)
