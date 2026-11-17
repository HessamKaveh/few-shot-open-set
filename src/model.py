import torch.nn as nn


def conv_block(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(),
        nn.MaxPool2d(2),
    )


class EmbeddingNet(nn.Module):
    """
    Network that maps each image to an embedding vector.
    In Prototypical Networks, distances between these embeddings measure class similarity.
    """
    def __init__(self, in_channels=1, hidden_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            conv_block(in_channels, hidden_dim),  # 28x28 -> 14x14
            conv_block(hidden_dim, hidden_dim),    # 14x14 -> 7x7
            conv_block(hidden_dim, hidden_dim),    # 7x7 -> 3x3
            conv_block(hidden_dim, hidden_dim),    # 3x3 -> 1x1
        )

    def forward(self, x):
        x = self.encoder(x)
        return x.flatten(1)  # (B, hidden_dim)
