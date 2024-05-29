import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':
    plt.figure(figsize=(8, 6))
    data = np.load('./000008.npy')
    # Define range and angular parameters based on your radar information
    min_range = 1.0  # meters (rr_min)
    max_range = 33.7  # meters (rr_max)
    min_angular = -60.0  # degrees (ra_min)
    max_angular = 60.0  # degrees (ra_max)

    # Create a plot with customized range and angular information
    plt.imshow(data, origin='lower', extent=[min_angular, max_angular, min_range, max_range], aspect='auto')

    plt.xlabel("Angular (degrees)")
    plt.ylabel("Range (meters)")
    plt.title("Range-Azimuth Map")
    # plt.savefig("2019_04_09_BMS1000_000000_0000.png")
    plt.show()
