import numpy as np
import torch
import torch.nn as nn
from einops.layers.torch import Rearrange

from models.radio_decoder import RODDecoder
from models.ssl_encoder import SSLEncoder
from models.semantic_depth_feature_extractor import SemanticDepthFeatureExtractor


def pretrained_encoder(checkpoint_path):
    encoder_q = SSLEncoder()
    pretrained_weights = torch.load(checkpoint_path, map_location=torch.device('cpu'))
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
    def __init__(self, num_class=3, fuse_semantic_depth_feature=False):
        super(RadarObjectDetector, self).__init__()
        self.fuse_semantic_depth_feature = fuse_semantic_depth_feature
        self.encoder = pretrained_encoder('/Users/yluo/Downloads/checkpoint_0019.pth.tar')
        self.decoder = RODDecoder(num_class)
        if self.fuse_semantic_depth_feature:
            self.semantic_depth_feature_extractor = SemanticDepthFeatureExtractor()
        self.feature_reshape = Rearrange('b (p1 p2) d -> b d p1 p2', p1=16, p2=16)
        self.channel_resize = nn.Conv2d(1280, 256, kernel_size=1, stride=1, padding=0)
        self.norm = nn.BatchNorm2d(256)

    def forward(self, x, semantic_depth_tensor=None):
        with torch.no_grad():
            x = self.encoder(x)
        x = self.feature_reshape(x)
        x = self.channel_resize(x)
        x = self.norm(x)
        if self.fuse_semantic_depth_feature:
            assert semantic_depth_tensor is not None, \
                "Semantic depth tensor should not be None when feature fusion is desired"
            semantic_depth_feature = self.semantic_depth_feature_extractor(semantic_depth_tensor)
            x = x + semantic_depth_feature  # Add the semantic depth feature to every channel of the radar frame feature
            x = self.norm(x)
        return self.decoder(x)


if __name__ == '__main__':
    model = RadarObjectDetector(fuse_semantic_depth_feature=True)
    test = torch.randn(1, 3, 224, 224)
    semantic_depth_tensor_test = np.load('../../models/semantic_depth.npy')
    semantic_depth_tensor_test = np.expand_dims(semantic_depth_tensor_test, 0)
    semantic_depth_tensor_test = torch.from_numpy(semantic_depth_tensor_test).to(torch.float32)
    model.eval()
    with torch.no_grad():
        output = model(test, semantic_depth_tensor_test)
        print(output.shape)
