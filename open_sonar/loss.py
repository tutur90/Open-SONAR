from torch import nn


def reduce_loss(loss, reduction='mean'):
    return loss.mean() if reduction == 'mean' else loss.sum() if reduction == 'sum' else loss

class CrossEntropyLoss(nn.Module):
    """NLL loss with label smoothing.
    """
    def __init__(self, label_smoothing=0.1, ignore_index=-100, reduction='mean'):
        """Constructor for the CrossEntropy module with label smoothing.
        :param smoothing: label smoothing factor
        """
        super(CrossEntropyLoss, self).__init__()
        self.confidence = 1.0 - label_smoothing
        self.smoothing = label_smoothing
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, x, target):
        logprobs = nn.functional.log_softmax(x, dim=-1)
        mask = target != self.ignore_index
        target = target[mask]
        logprobs = logprobs[mask]
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return reduce_loss(loss, self.reduction)
