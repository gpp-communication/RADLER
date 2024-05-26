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
        self.ln = nn.Linear(512, 256)
        self.reshape = Rearrange('b (p1 p2) -> b p1 p2', p1=16, p2=16)
        self.norm = nn.BatchNorm2d(1)

    def forward(self, x):
        x = self.conv(x)
        x = self.backbone(x)
        x = torch.squeeze(x, (2, 3))
        x = self.ln(x)
        x = self.reshape(x)
        x = torch.unsqueeze(x, dim=1)
        x = self.norm(x)
        return x


if __name__ == '__main__':
    model = SemanticDepthFeatureExtractor()
    data = np.load('semantic_depth.npy')
    data = np.expand_dims(data, 0)
    data = torch.from_numpy(data).to(torch.float32)
    model.eval()
    with torch.no_grad():
        output = model(data)
        print(output.shape)
