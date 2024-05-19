import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from networks.downstream import pretrained_encoder


if __name__ == '__main__':
    X = np.random.randn(20, 3, 224, 224)
    encoder = pretrained_encoder('/Users/yluo/Downloads/checkpoint_0019.pth.tar')
    encoder.eval()
    output = encoder(torch.from_numpy(X).to(torch.float32))
    print(output.shape)
    output = output.detach().numpy()
    n_samples, nx, ny = output.shape
    output = np.reshape(output, (n_samples, nx*ny))
    X_embedded = TSNE(n_components=2, perplexity=3).fit_transform(output)
    print(X_embedded.shape)
    plt.scatter(X_embedded[:, 0], X_embedded[:, 1])
    plt.show()
