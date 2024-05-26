import os

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
    radar_paths = []
    radar_folder_dir = '/Users/yluo/Pictures/test'
    for root, dirs, files in os.walk(radar_folder_dir):
        for file in files:
            radar_paths.append(os.path.join(root, file))
    radar_data = load_radar_data(radar_paths)
    encoder = pretrained_encoder('/Users/yluo/Downloads/checkpoint_0059.pth.tar')
    encoder.eval()
    with torch.no_grad():
        output = encoder(torch.from_numpy(radar_data).to(torch.float32))
    output = output.detach().numpy()
    n_samples, nx, ny = output.shape
    output = np.reshape(output, (n_samples, nx*ny))
    X_embedded = TSNE(n_components=2, perplexity=10).fit_transform(output)
    print(X_embedded.shape)
    # plot samples in different group
    plt.scatter(X_embedded[:, 0], X_embedded[:, 1])
    plt.show()
