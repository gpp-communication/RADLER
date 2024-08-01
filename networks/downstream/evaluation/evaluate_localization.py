import os
import json
import numpy as np

from networks.downstream.evaluation.evaluate_ols import read_gt_txt, read_sub_txt, compute_ols_dts_gts


def calculate_localization_error(sub_dets: dict, gt_dets:dict, num_frames: int, ols_thres):
    errors = [[], [], []]
    for i in range(num_frames):
        for c in range(3):  # pedestrian, cyclist, car
            olss = compute_ols_dts_gts(gt_dets, sub_dets, i, c)
            if len(olss):
                for j in range(olss.shape[0]):
                    for k in range (olss.shape[1]):
                        if olss[j][k] > ols_thres:
                            dt = sub_dets[i, c][j]
                            gt = gt_dets[i, c][k]
                            [dt_x, dt_y] = [dt['range'] * np.cos(dt['angle']), dt['range'] * np.sin(dt['angle'])]
                            [gt_x, gt_y] = [gt['range'] * np.cos(gt['angle']), gt['range'] * np.sin(gt['angle'])]
                            # Calculate squared differences
                            squared_diffs = np.power([dt_x - gt_x, dt_y - gt_y], 2)

                            # Calculate mean squared error
                            mse = np.sqrt(np.sum(squared_diffs))
                            errors[c].append(mse)
    pedestrian_error = np.sum(errors[0]) / len(errors[0]) if len(errors[0]) > 0 else 0
    cyclist_error = np.sum(errors[1]) / len(errors[1]) if len(errors[1]) > 0 else 0
    car_error = np.sum(errors[2]) / len(errors[2]) if len(errors[2]) > 0 else 0
    return pedestrian_error, cyclist_error, car_error


def evaluate_localization_error(data_path, submit_dir, truth_dir):
    with open('../configs/object_config.json', 'r') as f:
        object_config = json.load(f)
    sub_names = sorted(os.listdir(submit_dir))
    gt_names = sorted(os.listdir(truth_dir))
    assert len(sub_names) == len(gt_names), "missing submission files!"
    for sub_name, gt_name in zip(sub_names, gt_names):
        if sub_name != gt_name:
            raise AssertionError("wrong submission file names!")
    errors = {
        'pedestrian': [],
        'cyclist': [],
        'car': []
    }
    for seqid, (sub_name, gt_name) in enumerate(zip(sub_names, gt_names)):
        gt_path = os.path.join(truth_dir, gt_name)
        sub_path = os.path.join(submit_dir, sub_name)
        n_frame = len(os.listdir(os.path.join(data_path, gt_name.rstrip('.txt'), 'IMAGES_0')))
        gt_dets = read_gt_txt(gt_path, n_frame, object_config)
        sub_dets = read_sub_txt(sub_path, n_frame, object_config)
        pedestrian_error, cyclist_error, car_error = calculate_localization_error(sub_dets, gt_dets, n_frame, 0.9)
        errors['pedestrian'].append(pedestrian_error)
        errors['cyclist'].append(cyclist_error)
        errors['car'].append(car_error)

    print(errors)


if __name__ == '__main__':
    data_path = './test_localization/'
    submit_dir = './test_localization/prediction'
    truth_dir = './test_localization/gt'
    evaluate_localization_error(data_path, submit_dir, truth_dir)
