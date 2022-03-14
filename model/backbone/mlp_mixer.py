# -*- coding:utf-8 -*-
import torch
import torch.nn as nn

from ..config import MLPMixerConfig


class MLPBlock(nn.Module):
    def __init__(self, dim, expansion_factor, dropout=0.1):
        super(MLPBlock, self).__init__()
        self.in_mlp = nn.Linear(dim, dim * expansion_factor, bias=True)
        self.activate = nn.GELU()
        self.out_mlp = nn.Linear(dim * expansion_factor, dim, bias=False)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        nn.init.xavier_normal_(self.in_mlp.weight)
        nn.init.zeros_(self.in_mlp.bias)
        nn.init.xavier_normal_(self.out_mlp.weight)

    def forward(self, batch_data):
        batch_data = self.dropout1(self.activate(self.in_mlp(batch_data)))
        batch_data = self.dropout2(self.out_mlp(batch_data))
        return batch_data


class MLPMixerBlock(nn.Module):
    def __init__(self, num_patches, hidden_dim, expansion_factor=4, dropout=0.1):
        super(MLPMixerBlock, self).__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        self.mlp1 = MLPBlock(num_patches, expansion_factor, dropout)
        self.mlp2 = MLPBlock(hidden_dim, expansion_factor, dropout)

    def forward(self, batch_data):
        # batch_size, seq_len, hidden_dim
        residual = batch_data
        batch_data = self.norm1(batch_data)
        batch_data = self.mlp1(batch_data.permute(0, 2, 1))
        batch_data = residual + batch_data.permute(0, 2, 1)

        residual = batch_data
        batch_data = self.norm2(batch_data)
        batch_data = self.mlp2(batch_data)
        batch_data = residual + batch_data

        return batch_data


class MLPMixer(nn.Module):
    def __init__(self, seq_len, patch_size, n_channels, hidden_dim, num_layers=4, expansion_factor=4, dropout=0.1):
        super(MLPMixer, self).__init__()
        assert seq_len % patch_size == 0
        self.n_channels = n_channels
        self.num_patches = seq_len // patch_size
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim

        self.embedding = nn.Linear(self.n_channels * self.patch_size, hidden_dim, bias=False)

        self.encoders = nn.Sequential(
            *[MLPMixerBlock(self.num_patches, hidden_dim, expansion_factor, dropout) for _ in range(num_layers)]
        )

        nn.init.xavier_normal_(self.embedding.weight)

        self.norm = nn.LayerNorm(hidden_dim)

    def _pickup_patching(self, batch_data):
        # batch_size, n_channels, seq_len
        batch_size = batch_data.size(0)
        batch_data = batch_data.view(batch_size, self.n_channels, self.num_patches, self.patch_size)
        batch_data = batch_data.permute(0, 2, 1, 3)
        batch_data = batch_data.contiguous().view(batch_size, self.num_patches, self.n_channels * self.patch_size)
        return batch_data

    def forward(self, batch_data):
        batch_data = self._pickup_patching(batch_data)
        batch_data = self.embedding(batch_data)
        batch_data = self.encoders(batch_data)
        batch_data = self.norm(batch_data)
        batch_data = torch.mean(batch_data, dim=1)
        return batch_data

    def get_output_size(self):
        return self.hidden_dim


def mlp_mixer(model_name: str, config: MLPMixerConfig):
    # mixer_s_16
    attributes = model_name.split('_')
    scales = attributes[1]
    config.patch_size = int(attributes[2])
    if scales == 'es':
        # Extra Small
        config.num_layers = 2
        config.hidden_dim = 128
    elif scales == 'ms':
        # Medium Small
        config.num_layers = 4
        config.hidden_dim = 256
    elif scales == 's':
        # Small
        config.num_layers = 8
        config.hidden_dim = 512
    elif scales == 'b':
        # Base
        config.num_layers = 12
        config.hidden_dim = 768
    elif scales == "l":
        # Large
        config.num_layers = 24
        config.hidden_dim = 1024
    return MLPMixer(seq_len=config.seq_len, patch_size=config.patch_size, n_channels=config.n_channels,
                    hidden_dim=config.hidden_dim, num_layers=config.num_layers,
                    expansion_factor=config.expansion_factor,
                    dropout=config.dropout)
