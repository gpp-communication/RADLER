sites = ['Arcisstrasse1', 'Arcisstrasse2', 'Arcisstrasse3', 'Arcisstrasse4',
         'Arcisstrasse5', 'Gabelsbergerstrasse1', 'Gabelsbergerstrasse2']

splits = ['train', 'test1']

objects = {'car': 0, 'cyclist': 0, 'pedestrian': 0}


def read_line(lines: []):
    for line in lines:
        obj = line.split(' ')[-1].rstrip('\n')
        objects[obj] += 1


for split in splits:
    for site in sites:
        print(split, site)
        annotation_static = '/Users/yluo/Pictures/CRTUM_new/data_cluster_1_2/downstream/' + split + '/' + site + '/Annotations.txt'
        annotation_moving = '/Users/yluo/Pictures/CRTUM_new/data_cluster_1_2/downstream/' + split + '/' + site + '/Annotations_moving.txt'
        with open(annotation_static, 'r') as f:
            lines = f.readlines()
            read_line(lines)
        with open(annotation_moving, 'r') as f:
            lines = f.readlines()
            read_line(lines)

print(objects)
