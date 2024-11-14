import torch
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from data_tools.ssl.CRTUM_dataset import CRTUMDataset


class CRUWDataset(CRTUMDataset):
    def __getitem__(self, idx):
        img_path = self.df['images'][idx]
        radar_path = self.df['radar_frames'][idx]
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
    import torch
    import torch.nn as nn
    import torch.nn.functional as F


    class FullyTrainableUpsampling(nn.Module):
        def __init__(self, in_channels):
            super(FullyTrainableUpsampling, self).__init__()

            # First transposed convolution layer to increase spatial dimensions
            self.upsample1 = nn.ConvTranspose2d(
                in_channels=in_channels, out_channels=in_channels,
                kernel_size=3, stride=2, padding=1, output_padding=1
            )

            # Second transposed convolution layer to further increase dimensions
            self.upsample2 = nn.ConvTranspose2d(
                in_channels=in_channels, out_channels=in_channels,
                kernel_size=3, stride=2, padding=1, output_padding=1
            )

            # Final layer to reach exactly 224x224
            self.final_conv = nn.Conv2d(
                in_channels=in_channels, out_channels=in_channels,
                kernel_size=3, stride=1, padding=1
            )

        def forward(self, x):
            x = F.relu(self.upsample1(x))  # From 128x128 to ~256x256
            x = F.relu(self.upsample2(x))  # From ~256x256 to ~512x512
            x = self.final_conv(x)  # Final convolution for refinement (brings to exact 224x224)

            # Optionally, you can crop the result to (224, 224) if necessary
            x = x[:, :, :224, :224]  # Ensuring exact dimensions of 224x224
            return x


    # Example usage
    model = FullyTrainableUpsampling(3)
    input_image = torch.randn(1, 3, 128, 128)  # Batch size of 1
    output_image = model(input_image)
    print("Output shape:", output_image.shape)


