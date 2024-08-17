import torch
import numpy as np
from data_tools.ssl import CRUWDataset
from torch.utils.data import DataLoader
from torchvision.transforms import transforms


class DownstreamDataset(CRUWDataset):
    def __init__(self, root_dir: str, radar_transform=None):
        super().__init__(root_dir, None, radar_transform)

    def __getitem__(self, idx):
        img_path = self.df['images'][idx]
        radar_path = self.df['radar_frames'][idx]
        radar_frame = np.load(radar_path)
        radar_frame = np.expand_dims(radar_frame, 2)
        radar_frame = np.repeat(radar_frame, 3, 2)
        radar_frame = np.pad(radar_frame, ((48, 48), (48, 48), (0, 0)), 'constant')
        
        if self.radar_transform is not None:
            radar_frame = self.radar_transform(radar_frame)
            radar_frame = radar_frame.to(dtype=torch.float32)

        return img_path, radar_frame


if __name__ == '__main__':
    dataset = DownstreamDataset('../../datasets/test', transforms.ToTensor())
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4)
    for img_path, radar, semantic_depth, gt_conf in dataloader:
        print(img_path, radar.shape, semantic_depth.shape, gt_conf.shape)
