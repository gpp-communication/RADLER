import torch
import argparse
import torch.nn as nn
from torch.nn.modules.module import T

from models.radio_decoder import RODDecode
from models.ssl_encoder import RODEncodeCDC

parser = argparse.ArgumentParser(description='Radar Object Detection')
parser.add_argument('--pretrained-model', type=str, default='')


def pretrained_encoder(pretrained_model):
    encoder_q = RODEncodeCDC()
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
            self.encoder = RODEncodeCDC()
        self.decoder = RODDecode(num_class)
        self.sigmoid = nn.Sigmoid()
        for param in self.encoder.parameters():
            param.requires_grad = False

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        x = self.sigmoid(x)
        return x

    def train(self: T, mode: bool = True) -> T:
        super().train(mode)
        self.encoder.eval()
        return self


if __name__ == '__main__':
    args = parser.parse_args()
    use_noise_channel = False
    n_classes = 3
    model = RadarObjectDetector(pretrained_model="", mode='test')
    test_data = torch.randn(1, 3, 1, 128, 128)
    model.eval()
    criterion = nn.BCELoss()
    with torch.no_grad():
        output = model(test_data)
        loss = criterion(output, torch.rand(1, 3, 1, 128, 128))
        print(loss)
        print(output.shape)
