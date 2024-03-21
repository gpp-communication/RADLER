import torch
import torchvision
import torch.nn as nn
from torchvision.models import ViT_H_14_Weights
from torchvision.models.feature_extraction import create_feature_extractor


class RadioEncoder(nn.Module):
    def __init__(self, image_size=224):
        super(RadioEncoder, self).__init__()
        self.patch_size = 14
        # TODO: Add pretrained weights
        self.feature_extractor = create_feature_extractor(torchvision.models.vit_h_14(),
                                                          return_nodes={"encoder.ln": "features"})

    def forward(self, image):
        x = self.feature_extractor(image)
        x = x['features']
        x = x[:, 1:]
        return x


if __name__ == '__main__':
    radio_encoder = RadioEncoder()
    # the required input size of using pretrained weight is 224, which is close to the range-azimuth map?
    img = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        output = radio_encoder(img)
        print(output.shape)
