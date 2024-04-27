import torch
import requests
import torchvision
import numpy as np
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.models import ViT_H_14_Weights
from torchvision.models.feature_extraction import create_feature_extractor

from data_tools.ssl.CRUW_dataset import CRUWDataset


class SSLEncoder(nn.Module):
    def __init__(self, image_size=224):
        super(SSLEncoder, self).__init__()
        self.patch_size = 14
        # TODO: Add pretrained weights
        self.feature_extractor = create_feature_extractor(torchvision.models.vit_h_14(),
                                                          return_nodes={"encoder.ln": "features"})

    def forward(self, image):
        x = self.feature_extractor(image)
        x = x['features']
        x = x[:, 1:]
        return x


def image_transform():
    return ViT_H_14_Weights.IMAGENET1K_SWAG_LINEAR_V1.transforms()


def radar_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

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
    vision_encoder = SSLEncoder()
    data_loader = CRUW_dataloader('./datasets/CRUW', batch_size=1, image_transform=image_transform(), radar_frames_transform=radar_transform())
    with torch.no_grad():
        for i, (images, radar_frames) in enumerate(data_loader):
            # img_output = vision_encoder(images)
            radar_output = vision_encoder(radar_frames.to(dtype=torch.float32))
            # print(img_output.shape)
