import os

import torch
import numpy as np
import matplotlib.pyplot as plt
from einops.layers.torch import Rearrange

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
        radar_frame = radar_frame.to(torch.float32)
        radar_frame = radar_frame.detach().numpy()
        radar_frames[i, :, :, :] = radar_frame
    return radar_frames


if __name__ == '__main__':
    radar_paths = []
    radar_folder_dir = '/home/stud/luoyu/storage/user/luoyu/projects/Radio-Vision-CityGML/utilities/test_npy_different'
    for root, dirs, files in os.walk(radar_folder_dir):
        for file in files:
            radar_paths.append(os.path.join(root, file))
    radar_data = load_radar_data(radar_paths)
    encoder = pretrained_encoder('/home/stud/luoyu/storage/user/luoyu/projects/Radio-Vision-CityGML/logs/checkpoints/ssl/random-transform-0.8/training-64-1e-05-512/checkpoint_0159.pth.tar')
    encoder.eval()
    rearrange = Rearrange('b (p1 p2) d -> b d p1 p2', p1=16, p2=16)
    with torch.no_grad():
        feature_maps = encoder(torch.from_numpy(radar_data).to(torch.float32))
    feature_maps = rearrange(feature_maps)
    processed_feature_maps = []
    for i in range(feature_maps.shape[0]):
        feature_map = feature_maps[i]
        feature_map = torch.sum(feature_map, dim=0) / feature_map.shape[0]
        processed_feature_maps.append(feature_map.data.cpu().numpy())

    print("Processed feature maps shape")
    for fm in processed_feature_maps:
        print(fm.shape)
    np.save('feature_maps.npy', processed_feature_maps)
    # Plot the feature maps
    fig = plt.figure(figsize=(30, 50))
    for i in range(len(processed_feature_maps)):
        ax = fig.add_subplot(5, 4, i + 1)
        ax.imshow(processed_feature_maps[i])
        ax.axis("off")
        ax.set_title(i, fontsize=30)
    plt.show()
    plt.savefig('feature_maps.png')
