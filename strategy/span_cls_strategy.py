import torch
import torch.nn as nn


class SpanCLSStrategy(nn.Module):
    def __init__(self, model: nn.Module, head: nn.Module):
        super(SpanCLSStrategy, self).__init__()
        self.model = model
        self.head = head

        self.loss_fn = nn.CrossEntropyLoss(reduction='mean')

    def forward(self, accData, gyrData, label):
        data = torch.cat([accData, gyrData], dim=1)

        features = self.model(data)

        logits = self.head(features)

        return self.loss_fn(logits, label[:, 0])
