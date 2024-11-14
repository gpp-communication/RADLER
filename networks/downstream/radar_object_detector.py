import torch
import argparse
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.module import T
from einops.layers.torch import Rearrange

from models.radio_decoder import Decoder
from models.ssl_encoder import SSLEncoder

parser = argparse.ArgumentParser(description='Radar Object Detection')
parser.add_argument('--pretrained-model', type=str, default='')


def pretrained_encoder(pretrained_model):
    encoder_q = SSLEncoder()
    pretrained_weights = torch.load(pretrained_model, map_location=torch.device('cpu'))
    state_dict = pretrained_weights['state_dict']
    encoder_q_state_dict_old = {k: v for k, v in state_dict.items() if k.startswith('module.encoder_q.')}
    encoder_q_state_dict_new = {}
    for k, v in encoder_q_state_dict_old.items():
        new_k = k.replace('module.encoder_q.', '')
        encoder_q_state_dict_new[new_k] = encoder_q_state_dict_old[k]
    del encoder_q_state_dict_old
    del pretrained_weights
    encoder_q.load_state_dict(encoder_q_state_dict_new)
    return encoder_q


class RadarObjectDetector(nn.Module):
    def __init__(self, pretrained_model, mode, num_class=3):
        super(RadarObjectDetector, self).__init__()
        if mode == 'train':
            self.encoder = pretrained_encoder(pretrained_model)
        elif mode == 'test':
            self.encoder = SSLEncoder()
        self.upsample = nn.ConvTranspose2d(
            in_channels=num_class, out_channels=num_class,
            kernel_size=3, stride=2, padding=1, output_padding=1
        )
        self.decoder = Decoder(num_class)
        self.feature_reshape = Rearrange('b (p1 p2) d -> b d p1 p2', p1=16, p2=16)
        self.channel_resize = nn.Conv2d(1280, 256, kernel_size=1, stride=1, padding=0)
        self.norm = nn.BatchNorm2d(256)

        for param in self.encoder.parameters():
            param.requires_grad = False

    def forward(self, x):
        x = self.upsample(x)
        x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        x = self.encoder(x)
        x = self.feature_reshape(x)
        x = self.channel_resize(x)
        x = self.norm(x)
        return self.decoder(x)

    def train(self: T, mode: bool = True) -> T:
        super().train(mode)
        self.encoder.eval()
        return self


if __name__ == '__main__':
    args = parser.parse_args()
    use_noise_channel = False
    n_classes = 3
    model = RadarObjectDetector(pretrained_model="", mode='test')
    test_data = torch.randn(1, 3, 128, 128)
    model.eval()
    criterion = nn.BCELoss()
    with torch.no_grad():
        output = model(test_data)
        loss = criterion(output, torch.rand(1, 3, 128, 128))
        print(loss)
        print(output.shape)
