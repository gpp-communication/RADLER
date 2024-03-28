import math
import numpy as np
import scipy.constants


def get_class_id(class_str, classes):
    if class_str in classes:
        class_id = classes.index(class_str)
    else:
        if class_str == '':
            raise ValueError("No class name found")
        else:
            class_id = -1000
    return class_id


def find_nearest(array, value):
    """Find nearest value to 'value' in 'array'."""
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx, array[idx]


def ra2idx(rng, agl, range_grid, angle_grid):
    """Mapping from absolute range (m) and azimuth (rad) to ra indices."""
    rng_id, _ = find_nearest(range_grid, rng)
    agl_id, _ = find_nearest(angle_grid, agl)
    return rng_id, agl_id


def confmap2ra(name, radordeg='rad'):
    """
    Map confidence map to range(m) and angle(deg): not uniformed angle
    :param radar_configs: radar configurations
    :param name: 'range' for range mapping, 'angle' for angle mapping
    :param radordeg: choose from radius or degree for angle grid
    :return: mapping grids
    """
    # TODO: add more args for different network settings
    Fs = 4e6  # sample frequency
    sweepSlope = 21.0017e12
    num_crop = 3
    fft_Rang = 128 + 2 * num_crop
    fft_Ang = 128
    c = scipy.constants.speed_of_light

    if name == 'range':
        freq_res = Fs / fft_Rang
        freq_grid = np.arange(fft_Rang) * freq_res
        rng_grid = freq_grid * c / sweepSlope / 2  # freq_grid / sweepSlop = time_grid
        rng_grid = rng_grid[num_crop:fft_Rang - num_crop]
        return rng_grid

    if name == 'angle':
        # for [-90, 90], w will be [-1, 1]
        w = np.linspace(math.sin(math.radians(-90)),
                        math.sin(math.radians(90)),
                        128)
        if radordeg == 'deg':
            agl_grid = np.degrees(np.arcsin(w))  # rad to deg
        elif radordeg == 'rad':
            agl_grid = np.arcsin(w)  # -pi/2 - pi/2
        else:
            raise TypeError
        return agl_grid


def generate_confmap(n_obj, obj_info, config_dict, gaussian_thres=36):
    """
    Generate confidence map a radar frame.
    :param n_obj: number of objects in this frame
    :param obj_info: obj_info includes metadata information
    :param dataset: dataset object
    :param config_dict: rodnet configurations
    :param gaussian_thres: threshold for gaussian distribution in confmaps
    :return: generated confmap
    """
    n_class = 3
    classes = ['pedestrian', 'car', 'cyclist']
    confmap_sigmas = config_dict['confmap_cfg']['confmap_sigmas']
    confmap_sigmas_interval = config_dict['confmap_cfg']['confmap_sigmas_interval']
    confmap_length = config_dict['confmap_cfg']['confmap_length']
    range_grid = confmap2ra(name='range')
    angle_grid = confmap2ra(name='angle')

    confmap = np.zeros((n_class, 128, 128), dtype=float)
    for objid in range(n_obj):
        rng_idx, agl_idx = ra2idx(obj_info['centers'][objid][0], obj_info['centers'][objid][1], range_grid, angle_grid)
        print(rng_idx, agl_idx)
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
        for i in range(128):
            for j in range(128):
                distant = (((rng_idx - i) * 2) ** 2 + (agl_idx - j) ** 2) / sigma ** 2
                if distant < gaussian_thres:  # threshold for confidence maps
                    value = np.exp(- distant / 2) / (2 * math.pi)
                    confmap[class_id, i, j] = value if value > confmap[class_id, i, j] else confmap[class_id, i, j]

    return confmap


if __name__ == '__main__':
    obj_info = {'centers': [[12.0443, -0.0175], [11.5982, 0.3840]], 'categories': ['pedestrian', 'cyclist']}
    confmap_dict = {'confmap_cfg': dict(
        confmap_sigmas={
            'pedestrian': 15,
            'cyclist': 20,
            'car': 30,
            # 'van': 40,
            # 'truck': 50,
        },
        confmap_sigmas_interval={
            'pedestrian': [5, 15],
            'cyclist': [8, 20],
            'car': [10, 30],
            # 'van': [15, 40],
            # 'truck': [20, 50],
        },
        confmap_length={
            'pedestrian': 1,
            'cyclist': 2,
            'car': 3,
            # 'van': 4,
            # 'truck': 5,
        }
    )}
    print(generate_confmap(2, obj_info, confmap_dict))
    # print(confmap2ra('angle'))
