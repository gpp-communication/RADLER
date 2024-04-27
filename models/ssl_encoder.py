import torch
import requests
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from torchvision.models import ViT_H_14_Weights
from torchvision.models.feature_extraction import create_feature_extractor


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


def radar_transfrom():
    return transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


if __name__ == '__main__':
    vision_encoder = SSLEncoder()
    preprocess = ViT_H_14_Weights.IMAGENET1K_SWAG_LINEAR_V1.transforms()
    print(preprocess)
    img = Image.open(requests.get(
        "https://raw.githubusercontent.com/pytorch/ios-demo-app/master/HelloWorld/HelloWorld/HelloWorld/image.png",
        stream=True).raw)
    img = preprocess(img)
    img = img.unsqueeze(0)
    with torch.no_grad():
        output = vision_encoder(img)
        print(output.shape)
