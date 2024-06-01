import os
import numpy as np
import matplotlib.pyplot as plt


def plot_radar_data(radar_file_path: str):
    frame_number = os.path.basename(radar_file_path).replace('.npy', '')
    data = np.load(radar_file_path)
    ax.imshow(data, origin='lower', aspect='auto')

    plt.xlabel("Angular (degrees)")
    plt.ylabel("Range (meters)")
    plt.title("Range-Azimuth Map - No.%d" % int(frame_number))

    major_x_ticks = np.arange(0, 221, 10)
    minor_x_ticks = np.arange(0, 221, 5)
    major_y_ticks = np.arange(0, 224, 10)
    minor_y_ticks = np.arange(0, 224, 5)

    ax.set_xticks(major_x_ticks)
    ax.set_xticks(minor_x_ticks, minor=True)
    ax.set_yticks(major_y_ticks)
    ax.set_yticks(minor_y_ticks, minor=True)

    # And a corresponding grid
    ax.grid(which='minor', alpha=0.2)
    ax.grid(which='major', alpha=0.5)
    radar_png_folder = os.path.join(os.path.dirname(os.path.dirname(radar_data_folder)), 'radar_png')
    os.makedirs(radar_png_folder, exist_ok=True)
    plt.savefig(os.path.join(radar_png_folder, f'{frame_number}.png'))
    plt.cla()


if __name__ == '__main__':
    sites = ['Arcisstraße1', 'Arcisstraße2', 'Arcisstraße3', 'Arcisstraße4',
             'Arcisstraße5', 'Gabelsbergerstraße1', 'Gabelsbergerstraße2']
    fig, ax = plt.subplots(figsize=(8, 6))
    for split in ['training', 'testing']:
        for site in sites:
            print(split, site)
            radar_data_folder = ('/Users/yluo/Pictures/CRTUM/data_cluster_1_2/downstream/' + split + '/'
                                 + site + '/RADAR_RA_H/')
            for radar_file in os.listdir(radar_data_folder):
                plot_radar_data(os.path.join(radar_data_folder, radar_file))
