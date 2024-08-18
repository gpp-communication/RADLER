import os
import torch
import numpy as np
from data_tools.ssl import CRUWDataset
from torch.utils.data import DataLoader
from torchvision.transforms import transforms


class DownstreamDataset(CRUWDataset):
    def __init__(self, root_dir: str, radar_transform=None):
        super().__init__(root_dir, None, radar_transform)

        gt_confmaps = []
        for data_folder in self.data_folders:
            confmaps_folder = os.path.join(root_dir, data_folder, 'GT_CONFMAPS')
            for gt_confmap_file in sorted(os.listdir(confmaps_folder)):
                gt_confmaps.append(os.path.abspath(os.path.join(confmaps_folder, gt_confmap_file)))
        self.df['gt_confmaps'] = gt_confmaps

    def __getitem__(self, idx):
        img_path = self.df['images'][idx]
        radar_path = self.df['radar_frames'][idx]
        print(radar_path)
        gt_confmap_path = self.df['gt_confmaps'][idx]
        radar_frame = np.load(radar_path)
        radar_frame = np.sqrt(radar_frame[:, :, 0] ** 2 + radar_frame[:, :, 1] ** 2)
        radar_frame = np.expand_dims(radar_frame, 2)
        radar_frame = np.repeat(radar_frame, 3, 2)
        radar_frame = np.pad(radar_frame, ((48, 48), (48, 48), (0, 0)), 'constant')
        
        if self.radar_transform is not None:
            radar_frame = self.radar_transform(radar_frame)
            radar_frame = radar_frame.to(dtype=torch.float32)

        gt_confmap = np.load(gt_confmap_path)
        gt_confmap = torch.from_numpy(gt_confmap).float()
        return img_path, radar_frame, gt_confmap[:3]


if __name__ == '__main__':
    dataset = DownstreamDataset('/Users/yluo/Pictures/CRUW/downstream/sequences/train', transforms.ToTensor())
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4)
    for img_path, radar, gt_conf in dataloader:
        print(img_path, radar.shape, gt_conf.shape)
