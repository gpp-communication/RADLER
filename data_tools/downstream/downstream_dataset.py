import os
import torch
import numpy as np
from PIL import Image
from data_tools.ssl import CRTUMDataset


class DownstreamDataset(CRTUMDataset):
    def __init__(self, root_dir: str, img_transform=None, radar_transform=None, semantic_depth_transform=None,
                 semantic_depth_file_name='semantic_depth_35.npy'):
        super().__init__(root_dir, img_transform, radar_transform)
        self.semantic_depth_transform = semantic_depth_transform
        self.semantic_depth_file_name = semantic_depth_file_name

    def getitem(self, idx):
        img_path = self.df['images'][idx]
        radar_path = self.df['radar_frames'][idx]
        image = Image.open(img_path).convert('RGB')
        radar_frame = np.load(radar_path)
        radar_frame = np.expand_dims(radar_frame, 2)
        radar_frame = np.repeat(radar_frame, 3, 2)
        radar_frame = np.pad(radar_frame, ((0, 0), (2, 1), (0, 0)), 'constant')
        semantic_depth_tensor = np.load(os.path.join(os.path.commonpath([img_path, radar_path]),
                                                     self.semantic_depth_file_name))
        if self.image_transform is not None:
            image = self.image_transform(image)
        if self.radar_transform is not None:
            radar_frame = self.radar_transform(radar_frame)
            radar_frame = radar_frame.to(dtype=torch.float32)
        if self.semantic_depth_transform is not None:
            semantic_depth_tensor = self.semantic_depth_transform(semantic_depth_tensor)
            semantic_depth_tensor = semantic_depth_tensor.to(dtype=torch.float32)
        return image, radar_frame, semantic_depth_tensor
