import torch
import torch.nn as nn

from models.radio_encoder import RadioEncoder
from models.radio_decoder import RODDecoder


class ROD(nn.Module):
    def __init__(self, num_class=3):
        super(ROD, self).__init__()
        self.encoder = RadioEncoder()
        self.decoder = RODDecoder(num_class)

    def forward(self, x):
        return self.decoder(self.encoder(x))


if __name__ == '__main__':
    model = ROD()
    test = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        output = model(test)
        print(output.shape)
