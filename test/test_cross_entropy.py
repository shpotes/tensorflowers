import torch
import torch.nn as nn
from src.loss.cross_entropy import SparseCrossEntropyLoss

def test_sparse_cross_entropy():
    baseline_impl = nn.CrossEntropyLoss()
    custom_impl = SparseCrossEntropyLoss()

    logits = torch.randn(3, 5)
    target = torch.empty(3, dtype=torch.long).random_(5)

    torch_res = baseline_impl(logits, target)
    
    one_hot_target = nn.functional.one_hot(target, 5).float() 
    our_res = custom_impl(logits, one_hot_target)
    
    torch.testing.assert_close(torch_res, our_res)
