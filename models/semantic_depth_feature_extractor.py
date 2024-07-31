import torch
import torchvision
import numpy as np
import torch.nn as nn
from einops.layers.torch import Rearrange


class SemanticDepthFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 3, (1, 1), (1, 1))
        self.backbone = nn.Sequential(*(list(torchvision.models.resnet18().children())[:-1]))
        self.convt1 = nn.ConvTranspose2d(512, 256, kernel_size=(6, 6), stride=(2, 2), padding=(2, 2))
        self.convt2 = nn.ConvTranspose2d(256, 128, kernel_size=(6, 6), stride=(2, 2), padding=(2, 2))
        self.convt3 = nn.ConvTranspose2d(128, 128, kernel_size=(6, 6), stride=(2, 2), padding=(2, 2))
        self.convt4 = nn.ConvTranspose2d(128, 128, kernel_size=(6, 6), stride=(2, 2), padding=(2, 2))
        self.norm = nn.BatchNorm2d(128)

    def forward(self, x):
        x = self.conv(x)
        x = self.backbone(x)
        x = self.convt1(x)
        x = self.convt2(x)
        x = self.convt3(x)
        x = self.convt4(x)
        x = self.norm(x)
        return x


if __name__ == '__main__':
    model = SemanticDepthFeatureExtractor()
    data = np.load('semantic_depth.npy')
    data = np.transpose(data, [2, 0, 1])
    data = np.expand_dims(data, 0)
    data = torch.from_numpy(data).to(torch.float32)
    model.eval()
    with torch.no_grad():
        output = model(data)
        print(output.shape)
