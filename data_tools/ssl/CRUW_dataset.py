import torch
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from .CRTUM_dataset import CRTUMDataset


class CRUWDataset(CRTUMDataset):
    def __getitem__(self, idx):
        img_path = self.df['image'][idx]
        radar_path = self.df['radar_frame'][idx]
        image = Image.open(img_path).convert('RGB')
        radar_frame = np.load(radar_path)  # [128, 128, 2]: the radar data from CURW are in RI(Read Imaginary)
        radar_frame = np.sqrt(radar_frame[:, :, 0]**2 + radar_frame[:, :, 1]**2)
        radar_frame = np.expand_dims(radar_frame, 2)
        radar_frame = np.repeat(radar_frame, 3, 2)
        # radar_frame = np.transpose(radar_frame, (2, 0, 1))
        radar_frame = np.pad(radar_frame, ((48, 48), (48, 48), (0, 0)), 'constant')
        if self.image_transform is not None:
            image = self.image_transform(image)
        if self.radar_transform is not None:
            radar_frame = self.radar_transform(radar_frame)
        return image, radar_frame.to(dtype=torch.float32)


def CRUW_dataloader(root, batch_size, num_workers=4, image_transform=None,
                     radar_frames_transform=None, pin_memory=True):
    dataset = CRUWDataset(root, image_transform, radar_frames_transform)
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    return dataloader


if __name__ == '__main__':
    # CRUW_dataset = CRUWDataset('../../datasets/CRUW')
    img_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    radar_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    dataloader = CRUW_dataloader('../../datasets/CRUW', batch_size=32, num_workers=4, image_transform=img_transform,
                                  radar_frames_transform=radar_transform)
    for i, (images, radar_frames) in enumerate(dataloader):
        print(images.shape, radar_frames.shape)

