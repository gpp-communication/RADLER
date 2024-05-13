import os
import torch
import numpy as np
import pandas as pd
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


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
        self.df = pd.DataFrame(np.column_stack([images, radar_frames]), columns=['images', 'radar_frames'])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df['images'][idx]
        radar_path = self.df['radar_frames'][idx]
        image = Image.open(img_path).convert('RGB')  # TODO: check the RGB order in Image
        radar_frame = np.load(radar_path)
        radar_frame = np.expand_dims(radar_frame, 2)
        radar_frame = np.repeat(radar_frame, 3, 2)
        radar_frame = np.pad(radar_frame, ((0, 0), (2, 1), (0, 0)), 'constant')
        if self.image_transform is not None:
            image = self.image_transform(image)
        if self.radar_transform is not None:
            radar_frame = self.radar_transform(radar_frame)
            radar_frame = radar_frame.to(dtype=torch.float32)
        return image, radar_frame


def CRTUM_dataloader(root, batch_size, num_workers=4, image_transform=None,
                     radar_frames_transform=None, pin_memory=True):
    dataset = CRTUMDataset(root, image_transform, radar_frames_transform)
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    return dataloader


if __name__ == '__main__':
    img_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    radar_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    dataloader = CRTUM_dataloader('../../datasets/CRTUM/data_cluster_1_2/pretext', batch_size=32, num_workers=4, image_transform=img_transform,
                                  radar_frames_transform=radar_transform)
    for i, (images, radar_frames) in enumerate(dataloader):
        print(images.shape, radar_frames.shape, radar_frames.dtype)
        break
