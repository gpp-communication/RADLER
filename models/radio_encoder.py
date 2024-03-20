import torch
import torchvision
import torch.nn as nn
from einops.layers.torch import Rearrange
from torchvision.models import ViT_H_14_Weights
from torchvision.models.feature_extraction import create_feature_extractor


class RadioEncoder(nn.Module):
    def __init__(self, image_size=224):
        super(RadioEncoder, self).__init__()
        self.patch_size = 14
        # TODO: Add pretrained weights
        self.feature_extractor = create_feature_extractor(torchvision.models.vit_h_14(),
                                                          return_nodes={"encoder.ln": "features"})
        self.feature_reshape = Rearrange('b (p1 p2) d -> b d p1 p2', p1=image_size // self.patch_size,
                                         p2=image_size // self.patch_size)
        self.channel_resize = nn.Conv2d(torchvision.models.vit_h_14().hidden_dim, 256, kernel_size=1, stride=1,
                                        padding=0, bias=False)

    def forward(self, image):
        x = self.feature_extractor(image)
        x = x['features']
        x = x[:, 1:]
        x = self.feature_reshape(x)
        print(x.shape)
        x = self.channel_resize(x)
        return x


if __name__ == '__main__':
    radio_encoder = RadioEncoder()
    # the required input size of using pretrained weight is 224, which is close to the range-azimuth map?
    img = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        output = radio_encoder(img)
        print(output.shape)
