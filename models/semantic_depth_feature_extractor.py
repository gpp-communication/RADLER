import torch
import torchvision
import numpy as np
import torch.nn as nn


class SemanticDepthFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 3, (1, 1), (1, 1))
        self.backbone = nn.Sequential(*(list(torchvision.models.resnet18().children())[:-1]))
        self.ln = nn.Linear(512, 256)
        assert torch.__version__.split('.')[0] == '2', "Pytorch version has to be greater than or equal to 2.0"

    def forward(self, x):
        x = self.conv(x)
        x = self.backbone(x)
        x = torch.squeeze(x, (2, 3))
        x = self.ln(x)
        x = torch.reshape(x, (x.shape[0], 16, 16))
        x = torch.unsqueeze(x, dim=1)
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
