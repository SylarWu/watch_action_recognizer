# -*- coding:utf-8 -*-
import torch.nn as nn


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
    def __init__(self, seq_len, hidden_dim, expansion_factor=4, dropout=0.1):
        super(MLPMixerBlock, self).__init__()
        self.norm1 = nn.BatchNorm1d(hidden_dim)
        self.norm2 = nn.BatchNorm1d(hidden_dim)

        self.mlp1 = MLPBlock(seq_len, expansion_factor, dropout)
        self.mlp2 = MLPBlock(hidden_dim, expansion_factor, dropout)

    def forward(self, batch_data):
        # batch_size, hidden_dim, seq_len
        residual = batch_data
        batch_data = self.norm1(batch_data)
        batch_data = self.mlp1(batch_data)
        batch_data = residual + batch_data

        residual = batch_data
        batch_data = self.norm2(batch_data)
        batch_data = self.mlp2(batch_data.permute(0, 2, 1))
        batch_data = residual + batch_data.permute(0, 2, 1)

        return batch_data


class MLPMixer(nn.Module):
    def __init__(self, seq_len, n_channels, hidden_dim, num_layers=4, expansion_factor=4, dropout=0.1):
        super(MLPMixer, self).__init__()

        self.projection = nn.Linear(n_channels, hidden_dim, bias=False)

        self.encoders = nn.ModuleList(
            [MLPMixerBlock(seq_len, hidden_dim, expansion_factor, dropout) for _ in range(num_layers)]
        )

        nn.init.xavier_normal_(self.projection.weight)

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, batch_data):
        batch_data = self.dropout(self.projection(batch_data))
        residual = None
        for encoder in self.encoders:
            if residual is None:
                residual = batch_data
                batch_data = encoder(batch_data)
            else:
                residual = residual + batch_data
                batch_data = encoder(residual)
        return self.norm(residual + batch_data)
