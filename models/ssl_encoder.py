import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.models import ViT_H_14_Weights
from torchvision.models.feature_extraction import create_feature_extractor

from data_tools.ssl.CRUW_dataset import CRUWDataset
from models.u_net import UNet


class RODEncodeCDC(nn.Module):
    def __init__(self, in_channels=3):
        super(RODEncodeCDC, self).__init__()
        self.conv1a = nn.Conv3d(in_channels=in_channels, out_channels=64,
                                kernel_size=(9, 5, 5), stride=(1, 1, 1), padding=(4, 2, 2))
        self.conv1b = nn.Conv3d(in_channels=64, out_channels=64,
                                kernel_size=(9, 5, 5), stride=(2, 2, 2), padding=(4, 2, 2))
        self.conv2a = nn.Conv3d(in_channels=64, out_channels=128,
                                kernel_size=(9, 5, 5), stride=(1, 1, 1), padding=(4, 2, 2))
        self.conv2b = nn.Conv3d(in_channels=128, out_channels=128,
                                kernel_size=(9, 5, 5), stride=(2, 2, 2), padding=(4, 2, 2))
        self.conv3a = nn.Conv3d(in_channels=128, out_channels=256,
                                kernel_size=(9, 5, 5), stride=(1, 1, 1), padding=(4, 2, 2))
        self.conv3b = nn.Conv3d(in_channels=256, out_channels=256,
                                kernel_size=(9, 5, 5), stride=(1, 2, 2), padding=(4, 2, 2))
        self.bn1a = nn.BatchNorm3d(num_features=64)
        self.bn1b = nn.BatchNorm3d(num_features=64)
        self.bn2a = nn.BatchNorm3d(num_features=128)
        self.bn2b = nn.BatchNorm3d(num_features=128)
        self.bn3a = nn.BatchNorm3d(num_features=256)
        self.bn3b = nn.BatchNorm3d(num_features=256)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.bn1a(self.conv1a(x)))  # (B, 3, W, 128, 128) -> (B, 64, W, 128, 128)
        x = self.relu(self.bn1b(self.conv1b(x)))  # (B, 64, W, 128, 128) -> (B, 64, W/2, 64, 64)
        x = self.relu(self.bn2a(self.conv2a(x)))  # (B, 64, W/2, 64, 64) -> (B, 128, W/2, 64, 64)
        x = self.relu(self.bn2b(self.conv2b(x)))  # (B, 128, W/2, 64, 64) -> (B, 128, W/4, 32, 32)
        x = self.relu(self.bn3a(self.conv3a(x)))  # (B, 128, W/4, 32, 32) -> (B, 256, W/4, 32, 32)
        x = self.relu(self.bn3b(self.conv3b(x)))  # (B, 256, W/4, 32, 32) -> (B, 256, W/4, 16, 16)
        return x


class SSLEncoder(nn.Module):
    def __init__(self, input_data='radar'):
        super(SSLEncoder, self).__init__()
        self.patch_size = 14
        self.input_data = input_data
        self.upsample = UNet(in_channels=3, out_channels=3, kernel_size=3,
                             padding=1, stride=1)
        self.feature_extractor = create_feature_extractor(
            torchvision.models.vit_h_14(weights=ViT_H_14_Weights.IMAGENET1K_SWAG_LINEAR_V1),
            return_nodes={"encoder.ln": "features"}
        )

    def forward(self, data):
        if self.input_data == 'radar':
            x = self.upsample(data)
            x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
            x = self.feature_extractor(x)
            x = x['features']
            x = x[:, 1:]
            return x
        elif self.input_data == 'image':
            x = self.feature_extractor(data)
            x = x['features']
            x = x[:, 1:]
            return x


def image_transform():
    return transforms.Compose([
        transforms.Resize([128, 128]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


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
    vision_encoder = RODEncodeCDC()
    rodnet_encoder = RODEncodeCDC()
    data_loader = CRUW_dataloader('../datasets/CRUW-test', batch_size=1, image_transform=image_transform(),
                                  radar_frames_transform=radar_transform())
    with torch.no_grad():
        for i, (images, radar_frames) in enumerate(data_loader):
            img_output = vision_encoder(images)
            radar_output = rodnet_encoder(radar_frames.to(dtype=torch.float32))
            print(img_output.shape, radar_output.shape)
