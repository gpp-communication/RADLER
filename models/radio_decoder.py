import torch
import torch.nn as nn


class RODDecoder(nn.Module):

    def __init__(self, n_class):
        super(RODDecoder, self).__init__()
        self.convt1 = nn.ConvTranspose2d(in_channels=257, out_channels=128,
                                         kernel_size=(6, 6), stride=(2, 2), padding=(2, 2))
        self.convt2 = nn.ConvTranspose2d(in_channels=128, out_channels=64,
                                         kernel_size=(6, 6), stride=(2, 2), padding=(2, 2))
        self.convt3 = nn.ConvTranspose2d(in_channels=64, out_channels=32,
                                         kernel_size=(6, 6), stride=(2, 2), padding=(2, 2))
        self.convt4 = nn.ConvTranspose2d(in_channels=32, out_channels=16,
                                         kernel_size=(6, 6), stride=(2, 2), padding=(2, 2))
        self.conv = nn.Conv2d(in_channels=16, out_channels=n_class,
                              kernel_size=1)
        self.downsample = nn.Upsample(size=(224, 221), mode='bilinear')
        self.prelu = nn.PReLU()
        self.sigmoid = nn.Sigmoid()
        # self.upsample = nn.Upsample(size=(rodnet_configs['win_size'], radar_configs['ramap_rsize'],
        #                                   radar_configs['ramap_asize']), mode='nearest')

    def forward(self, x):
        x = self.prelu(self.convt1(x))  # (B, 256, 16, 16) -> (B, 128, 32, 32)
        x = self.prelu(self.convt2(x))  # (B, 128, 32, 32) -> (B, 64, 64, 64)
        x = self.prelu(self.convt3(x))  # (B, 64, 64, 64) -> (B, 32, 128, 128)
        x = self.prelu(self.convt4(x))  # (B, 32, 128, 128) -> (B, 16, 256, 256)
        x = self.conv(x)                # (B, 16, 256, 256) -> (B, 3, 256, 256)
        x = self.downsample(x)            # (B, 3, 256, 256) -> (B, 3, 224, 221)
        x = self.sigmoid(x)
        return x


if __name__ == '__main__':
    rod_decoder = RODDecoder(3)
    test = torch.randn(1, 256, 16, 16)
    with torch.no_grad():
        output = rod_decoder(test)
        print(output.shape)
