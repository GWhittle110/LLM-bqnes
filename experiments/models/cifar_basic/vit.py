"""
Vision transformer model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import einops.layers.torch
from einops import repeat
from experiments.trainers.torchTrain import torch_train
import experiments.datasets.cifar10 as dataset_module


class VIT(nn.Module):
    """
    Vision transformer class for 32x32 image
    """
    def __init__(self, trained=True):
        super(VIT, self).__init__()
        channels = 3
        transformer_dim = 128
        depth = 3
        heads = 8
        num_classes = 10
        dropout = 0.2
        image_size = 32
        patch_size = 4
        mlp_dim = 50

        num_patches = (image_size // patch_size) ** 2
        patch_dim = channels * patch_size**2
        self.to_patch_embedding = nn.Sequential(
            einops.layers.torch.Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, transformer_dim),
            nn.LayerNorm(transformer_dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, transformer_dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, transformer_dim))
        self.dropout = nn.Dropout(dropout)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=transformer_dim, nhead=heads)
        self.transformer = nn.TransformerEncoder(self.encoder_layer, num_layers=depth)
        self.mlp_head = nn.Sequential(
            nn.Linear(transformer_dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, num_classes))
        if trained:
            self.load_state_dict(torch.load('C:\\Users\\gwhit\PycharmProjects\\4YP\experiments\\models\\cifar_basic\\states\\vit.pth'))

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim=1)

        x = self.mlp_head(x)
        return F.softmax(x)


if __name__ == "__main__":
    model = VIT(trained=False)
    train_dataset = getattr(dataset_module, "train_dataset")
    test_dataset = getattr(dataset_module, "test_dataset")
    torch_train(model, train_dataset, test_dataset, "vit", device=torch.device("cuda:0"), learning_rate=0.01, momentum=0.9, batch_size_train=64, gamma=0.9, n_epochs=20)

