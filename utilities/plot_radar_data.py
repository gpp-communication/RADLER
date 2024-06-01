import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':
    fig, ax = plt.subplots(figsize=(8, 6))
    data = np.load('/Users/yluo/Pictures/CRTUM/data_cluster_1_2/downstream/training/Arcisstraße1/RADAR_RA_H/000431.npy')
    # Define range and angular parameters based on your radar information
    min_range = 1.0  # meters (rr_min)
    max_range = 33.7  # meters (rr_max)
    min_angular = -60.0  # degrees (ra_min)
    max_angular = 60.0  # degrees (ra_max)

    # Create a plot with customized range and angular information
    ax.imshow(data, origin='lower', aspect='auto')

    plt.xlabel("Angular (degrees)")
    plt.ylabel("Range (meters)")
    plt.title("Range-Azimuth Map")
    # Major ticks every 20, minor ticks every 5
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
    # plt.savefig("2019_04_09_BMS1000_000000_0000.png")
    plt.show()
