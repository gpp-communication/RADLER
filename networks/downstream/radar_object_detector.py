import torch
import torch.nn as nn
from einops.layers.torch import Rearrange

from models.radio_decoder import RODDecoder
from models.ssl_encoder import SSLEncoder


class RadarObjectDetector(nn.Module):
    def __init__(self, num_class=3):
        super(RadarObjectDetector, self).__init__()
        self.encoder = SSLEncoder()
        # self.encoder.load_state_dict(torch.load("models/ssl_encoder.pth"))  # TODO: load pretrained weights
        self.decoder = RODDecoder(num_class)
        self.feature_reshape = Rearrange('b (p1 p2) d -> b d p1 p2', p1=16, p2=16)
        self.channel_resize = nn.Conv2d(1280, 256, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        with torch.no_grad():
            x = self.encoder(x)
        x = self.feature_reshape(x)
        x = self.channel_resize(x)
        return self.decoder(x)


if __name__ == '__main__':
    model = RadarObjectDetector()
    test = torch.randn(1, 3, 224, 224)
    model.eval()
    with torch.no_grad():
        output = model(test)
        print(output.shape)
