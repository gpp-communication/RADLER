import torch
import torch.nn as nn
import torchvision

img = torch.randn(1, 3, 224, 224)

model = torchvision.models.vit_b_16()
feature_extractor = nn.Sequential(*list(model.children())[:-1])
# print(model)

# This is supposed to be the PREPROCESS
# But it is not done correctly, since the reshaping and permutation is not done
# Only the concolution
conv = feature_extractor[0]

# -> print(conv(img).shape)
# -> torch.Size([1, 768, 14, 14])
# This is not the desired output after preprocessing the image into
# flat patches. Also in the pytorch implementation, the class token
# and positional embedding are done extra on the forward method.

# This is the whole encoder sequence
encoder = feature_extractor[1]

# The MLP head at the end is gone, since you only selected the children until -1
# mlp = feature_extractor[2]

# This is how the model preprocess the image.
# The output shape is the one desired
x = model._process_input(img)

# -> print(x.shape)
# -> torch.Size([1, 197, 768])
# This is Batch x N_Patches+Class_Token x C * H_patch * W_patch
# Meaning   1   x   14*14  +     1      x 3 * 16* 16

# However, if you actually print the shape in here you only get 196 in dim=1
# This means that the class token in missing
# The positional_embedding is done inside the encoder, so I guess should be fine

# The next code is just copy paste from the forward method in the source code
# for the vit_b_16 from pytorch in order to get the

n = x.shape[0]
# Expand the class token to the full batch
batch_class_token = model.class_token.expand(n, -1, -1)
x = torch.cat([batch_class_token, x], dim=1)
x = encoder(x)

# Classifier "token" as used by standard language architectures
print(x.shape)
x = x[:, 0]
print(x.shape)
# Here you can use your own nn.Linear to map to your number of classes