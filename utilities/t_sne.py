import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from networks.downstream import pretrained_encoder
from models.ssl_encoder import radar_transform


def load_radar_data(paths):
    radar_frames = np.zeros((len(paths), 3, 224, 224))
    radar_trans = radar_transform()
    for i, path in enumerate(paths):
        radar_frame = np.load(path)
        radar_frame = np.expand_dims(radar_frame, 2)
        radar_frame = np.repeat(radar_frame, 3, 2)
        radar_frame = np.pad(radar_frame, ((0, 0), (2, 1), (0, 0)), 'constant')
        radar_frame = radar_trans(radar_frame)
        radar_frame = radar_frame.detach().numpy()
        radar_frames[i, :, :, :] = radar_frame
    return radar_frames


if __name__ == '__main__':
    radar_paths = [
        '/Users/yluo/Pictures/CRTUM/data_cluster_1_2/downstream/Arcisstraße1/RADAR_RA_H/000008.npy',
        '/Users/yluo/Pictures/CRTUM/data_cluster_1_2/downstream/Arcisstraße1/RADAR_RA_H/000009.npy',
        '/Users/yluo/Pictures/CRTUM/data_cluster_1_2/downstream/Arcisstraße1/RADAR_RA_H/000018.npy',
        '/Users/yluo/Pictures/CRTUM/data_cluster_1_2/downstream/Arcisstraße1/RADAR_RA_H/000019.npy',
        '/Users/yluo/Pictures/CRTUM/data_cluster_1_2/downstream/Arcisstraße1/RADAR_RA_H/000020.npy',
        '/Users/yluo/Pictures/CRTUM/data_cluster_1_2/downstream/Arcisstraße1/RADAR_RA_H/000021.npy',
        '/Users/yluo/Pictures/CRTUM/data_cluster_1_2/downstream/Arcisstraße1/RADAR_RA_H/000023.npy',
        '/Users/yluo/Pictures/CRTUM/data_cluster_1_2/downstream/Arcisstraße1/RADAR_RA_H/000024.npy',
        '/Users/yluo/Pictures/CRTUM/data_cluster_1_2/downstream/Arcisstraße1/RADAR_RA_H/000025.npy',
        '/Users/yluo/Pictures/CRTUM/data_cluster_1_2/downstream/Arcisstraße1/RADAR_RA_H/000026.npy',
        '/Users/yluo/Pictures/CRTUM/data_cluster_1_2/downstream/Arcisstraße1/RADAR_RA_H/000027.npy',
        '/Users/yluo/Pictures/CRTUM/data_cluster_1_2/downstream/Arcisstraße1/RADAR_RA_H/000030.npy',
        '/Users/yluo/Pictures/CRTUM/data_cluster_1_2/downstream/Arcisstraße1/RADAR_RA_H/000031.npy',
        '/Users/yluo/Pictures/CRTUM/data_cluster_1_2/downstream/Arcisstraße1/RADAR_RA_H/000032.npy',
        '/Users/yluo/Pictures/CRTUM/data_cluster_1_2/downstream/Arcisstraße1/RADAR_RA_H/000033.npy',
        '/Users/yluo/Pictures/CRTUM/data_cluster_1_2/downstream/Arcisstraße1/RADAR_RA_H/000034.npy',
        '/Users/yluo/Pictures/CRTUM/data_cluster_1_2/downstream/Arcisstraße1/RADAR_RA_H/000035.npy'
    ]
    radar_data = load_radar_data(radar_paths)
    encoder = pretrained_encoder('/Users/yluo/Downloads/checkpoint_0019.pth.tar')
    encoder.eval()
    output = encoder(torch.from_numpy(radar_data).to(torch.float32))
    output = output.detach().numpy()
    n_samples, nx, ny = output.shape
    output = np.reshape(output, (n_samples, nx*ny))
    X_embedded = TSNE(n_components=2, perplexity=3).fit_transform(output)
    print(X_embedded.shape)
    # plot samples in different group
    plt.scatter(X_embedded[:3, 0], X_embedded[:3, 1])
    plt.scatter(X_embedded[3:6, 0], X_embedded[3:6, 1])
    plt.scatter(X_embedded[6:, 0], X_embedded[6:, 1])
    plt.show()
