import numpy as np
import torch
import torch.nn as nn
from src.loss import SparseCrossEntropyLoss
from src.evaluation import CrossEntropyMetric
import tensorflow as tf 
from tensorflow.keras.metrics import CategoricalCrossentropy

def test_sparse_cross_entropy():
    baseline_impl = nn.CrossEntropyLoss()
    custom_impl = SparseCrossEntropyLoss()

    logits = torch.randn(3, 5)
    target = torch.empty(3, dtype=torch.long).random_(5)

    torch_res = baseline_impl(logits, target)
    
    one_hot_target = nn.functional.one_hot(target, 5).float() 
    our_res = custom_impl(logits, one_hot_target)
    
    torch.testing.assert_close(torch_res, our_res)

def test_cross_entropy_metric():
    tf_impl = CategoricalCrossentropy(from_logits=True)
    ours = CrossEntropyMetric()

    for _ in range(10):
        logits = torch.randn(3, 5)
        target = torch.empty(3, dtype=torch.long).random_(5)
        target = nn.functional.one_hot(target, 5).float()

        tf_impl.update_state(
            tf.convert_to_tensor(target.numpy()),
            tf.convert_to_tensor(logits.numpy())
        )

        ours.update(logits, target)

    np.testing.assert_allclose(
        tf_impl.result().numpy(),
        ours.compute().numpy()
    )