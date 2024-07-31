import json

from networks.downstream.confidence_map.generate_grids import confmap2ra


def idx2polar(annotation_file):
    with open('../networks/downstream/configs/radar_config.json') as radar_json:
        radar_configs = json.load(radar_json)
    range_grids = confmap2ra('range', radar_configs)
    angle_grids = confmap2ra('angle', radar_configs)
    objects = []
    with open(annotation_file, 'r') as f:
        for line in f.readlines():
            frame_no, range, angle, obj_class = line.split(' ')
            obj_dict = {'frame_no': int(frame_no), 'range': range_grids[int(range)],
                        'angle': angle_grids[int(angle)], 'object_class': obj_class.rstrip('\n')}
            objects.append(obj_dict)

    with open(annotation_file.replace('.txt', '_polar.txt'), 'w') as f:
        for obj in objects:
            f.write('%s %.4f %.4f %s\n' % (obj['frame_no'], obj['range'], obj['angle'], obj['object_class']))


def idx2polar_res(res_file):
    with open('../networks/downstream/configs/radar_config.json') as radar_json:
        radar_configs = json.load(radar_json)
    range_grids = confmap2ra('range', radar_configs)
    angle_grids = confmap2ra('angle', radar_configs)
    objects = []
    with open(res_file, 'r') as f:
        for line in f.readlines():
            frame_no, obj_class, range, angle, confidence_value = line.split(' ')
            obj_dict = {'frame_no': int(frame_no), 'range': range_grids[int(range)],
                        'angle': angle_grids[int(angle)], 'object_class': obj_class.rstrip('\n'),
                        'confidence_value': float(confidence_value)}
            objects.append(obj_dict)

    with open(res_file, 'w') as f:
        for obj in objects:
            f.write('%s %.4f %.4f %s %.4f\n' % (
            obj['frame_no'], obj['range'], obj['angle'], obj['object_class'], obj['confidence_value']))


if __name__ == '__main__':
    splits = ['train', 'test1']
    sites = ['Arcisstrasse1', 'Arcisstrasse2', 'Arcisstrasse3', 'Arcisstrasse4',
             'Arcisstrasse5', 'Gabelsbergerstrasse1', 'Gabelsbergerstrasse2']
    # for split in splits:
    #     for site in sites:
    #         print(split, site)
    #         idx2polar(
    #             '/Users/yluo/Pictures/CRTUM_new/data_cluster_1_2/downstream/' + split + '/' + site + '/Annotations_moving.txt')
    #         idx2polar(
    #             '/Users/yluo/Pictures/CRTUM_new/data_cluster_1_2/downstream/' + split + '/' + site + '/Annotations.txt')

    for site in sites:
        idx2polar_res('/Users/yluo/Downloads/res/rod-res/' + site + '.txt')
