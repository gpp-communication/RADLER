import torch
import torch.nn as nn
from datasets import load_dataset
from einops.layers.torch import Rearrange
from transformers import AutoImageProcessor, AutoModel


def dino_feature_extractor(image: torch.Tensor):
    processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
    model = AutoModel.from_pretrained('facebook/dinov2-base')

    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        features = outputs.last_hidden_state
    return features


class VisionEncoder(nn.Module):
    def __init__(self):
        super(VisionEncoder, self).__init__()
        self.feature_extractor = dino_feature_extractor
        self.feature_reshape = Rearrange('b (p1 p2) d -> b d p1 p2', p1=16, p2=16)
        self.channel_resize = nn.Conv2d(768, 256, kernel_size=1, stride=1, padding=0)

    def forward(self, image: torch.Tensor):
        features = self.feature_extractor(image)
        features = features[:, 1:]
        features = self.feature_reshape(features)
        features = self.channel_resize(features)
        return features


if __name__ == '__main__':
    vision_encoder = VisionEncoder()
    dataset = load_dataset("huggingface/cats-image", trust_remote_code=True)
    test_img = dataset["test"]["image"][0]
    output = vision_encoder(test_img)
    print(output, output.shape)  # [1, 256, 16, 16]
