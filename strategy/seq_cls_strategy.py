import torch.nn as nn


class SequenceCLSStrategy(nn.Module):
    def __init__(self, model: nn.Module, head: nn.Module):
        super(SequenceCLSStrategy, self).__init__()
        self.model = model
        self.head = head

        self.loss_fn = nn.CrossEntropyLoss(reduction='mean')

    def forward(self, accData, gyrData, label):
        features = self.model(accData, gyrData)

        logits = self.head(features.permute(0, 2, 1))

        batch_size = label.size(0)
        seq_len = label.size(1)
        label -= 1

        return self.loss_fn(logits.view((batch_size * seq_len, -1)), label.view((batch_size * seq_len)))
