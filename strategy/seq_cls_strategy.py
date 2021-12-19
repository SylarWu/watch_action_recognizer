import torch
import torch.nn as nn

class SequenceCLSStrategy(nn.Module):
    def __init__(self, model: nn.Module, head: nn.Module):
        super(SequenceCLSStrategy, self).__init__()
        self.model = model
        self.head = head

        self.loss_fn = nn.CrossEntropyLoss(reduction='mean')

    def forward(self, accData, gyrData, label):
        features = self.model(accData, gyrData)

        logits = self.head(features)

        return self.loss_fn(logits, label)