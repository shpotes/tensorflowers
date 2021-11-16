import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import reduce

class SparseCrossEntropyLoss(nn.Module):
    def forward(self, logit, target):
        # logit:  [N, d]
        # target: [N, d]

        target /= reduce(target, "b d -> b 1", "sum")
        cross_entropy = torch.mean(
            -torch.sum(F.log_softmax(logit, dim=1) * target, dim=1)
        )

        return cross_entropy