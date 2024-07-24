import torch
import torch.nn as nn


class DecoderSD(nn.Module):

    def __init__(self, n_class):
        super(DecoderSD, self).__init__()
        self.convt1 = nn.ConvTranspose2d(in_channels=384, out_channels=192,
                                         kernel_size=(6, 6), stride=(2, 2), padding=(2, 2))
        self.convt2 = nn.ConvTranspose2d(in_channels=192, out_channels=96,
                                         kernel_size=(6, 6), stride=(2, 2), padding=(2, 2))
        self.convt3 = nn.ConvTranspose2d(in_channels=96, out_channels=48,
                                         kernel_size=(6, 6), stride=(2, 2), padding=(2, 2))
        self.convt4 = nn.ConvTranspose2d(in_channels=48, out_channels=24,
                                         kernel_size=(6, 6), stride=(2, 2), padding=(2, 2))
        self.conv1 = nn.Conv2d(in_channels=24, out_channels=12, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels=12, out_channels=n_class, kernel_size=1)
        self.downsample = nn.Upsample(size=(224, 221), mode='bilinear')
        self.prelu = nn.PReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.prelu(self.convt1(x))  # (B, x, 16, 16) -> (B, x, 32, 32)
        x = self.prelu(self.convt2(x))  # (B, x, 32, 32) -> (B, x, 64, 64)
        x = self.prelu(self.convt3(x))  # (B, x, 64, 64) -> (B, x, 128, 128)
        x = self.prelu(self.convt4(x))  # (B, x, 128, 128) -> (B, x, 256, 256)
        x = self.conv1(x)               # (B, x, 256, 256) -> (B, x, 256, 256)
        x = self.conv2(x)               # (B, x, 256, 256) -> (B, 3, 256, 256)
        x = self.downsample(x)          # (B, 3, 256, 256) -> (B, 3, 224, 221)
        x = self.sigmoid(x)
        return x


if __name__ == '__main__':
    rod_decoder = DecoderSD(3)
    test = torch.randn(1, 256, 16, 16)
    with torch.no_grad():
        output = rod_decoder(test)
        print(output.shape)
