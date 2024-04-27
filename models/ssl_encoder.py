import torch
import requests
import torchvision
import torch.nn as nn
from PIL import Image
from torchvision.models import ViT_H_14_Weights
from torchvision.models.feature_extraction import create_feature_extractor


class VisionEncoder(nn.Module):
    def __init__(self, image_size=224):
        super(VisionEncoder, self).__init__()
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
    vision_encoder = VisionEncoder()
    preprocess = ViT_H_14_Weights.IMAGENET1K_SWAG_LINEAR_V1.transforms()
    print(preprocess)
    img = Image.open(requests.get("https://raw.githubusercontent.com/pytorch/ios-demo-app/master/HelloWorld/HelloWorld/HelloWorld/image.png", stream=True).raw)
    img = preprocess(img)
    img = img.unsqueeze(0)
    with torch.no_grad():
        output = vision_encoder(img)
        print(output.shape)
