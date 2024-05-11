import os
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class CRTUMDataset(Dataset):
    def __init__(self, root, img_transform=None, radar_transform=None):
        self.data_folders = os.listdir(root)
        self.image_transform = img_transform
        self.radar_transform = radar_transform
        images = []
        radar_frames = []
        for data_folder in self.data_folders:
            images_folder = os.path.join(root, data_folder, 'IMAGES_0')
            radar_frames_folder = os.path.join(root, data_folder, 'RADAR_RA_H')
            for image_file in os.listdir(images_folder):
                images.append(os.path.join(images_folder, image_file))
            for radar_file in os.listdir(radar_frames_folder):
                radar_frames.append(os.path.join(radar_frames_folder, radar_file))
        assert len(images) == len(radar_frames), "Number of images doesn't match number of radar frames"
        self.df = pd.DataFrame(np.column_stack([images, radar_frames]), columns=['image', 'radar_frame'])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df['images'][idx]
        radar_path = self.df['radar_frames'][idx]
        image = Image.open(img_path).convert('RGB')  # TODO: check the RGB order in Image
        radar = np.load(radar_path)
        if self.image_transform is not None:
            image = self.image_transform(image)  # TODO: to tensor
        if self.radar_transform is not None:
            radar = self.radar_transform(radar)  # TODO: to tensor
        return image, radar.to(dtype=torch.float32)


if __name__ == '__main__':
    dataset = CRTUMDataset('../../datasets/CRTUM/data_cluster_1_2/pretext')
    print(len(dataset))
