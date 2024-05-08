import torch
import torchvision
import torch.nn as nn


class SemanticDepthFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(*(list(torchvision.models.resnet18().children())[:-1]))

    def forward(self, x):
        return self.backbone(x)


if __name__ == '__main__':
    model = SemanticDepthFeatureExtractor()
    data = torch.randn(1, 3, 224, 224)
    model.eval()
    with torch.no_grad():
        output = model(data)
        print(output.shape)
