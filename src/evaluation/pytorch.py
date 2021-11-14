from typing import Any, Callable, Optional, Tuple 

import torch
from torch import Tensor
from torchmetrics.metric import Metric
from torchmetrics.utilities.checks import _check_same_shape
from torchmetrics.utilities.data import dim_zero_cat
from torch.nn import functional as F

def _cross_entropy_update(
    targets: Tensor, 
    logits: Tensor, 
) -> Tuple[Tensor, int]:
    """Updates and returns cross entropy scores for each observation and the total number of observations. 
    Checks same shape and 2D nature of the input tensors else raises ValueError.
    Args:
        targets: one hot - target tensors with shape ``[N, d]``
        logits: softmaxed model logits ``[N, d]``
    """
    _check_same_shape(targets, logits)
    if targets.ndim != 2 or logits.ndim != 2:
        raise ValueError(f"Expected both targets and logits to be 2D but got {targets.ndim} and {logits.ndim} respectively")

    total = targets.shape[0]

    measure = -torch.sum(F.log_softmax(logits, dim=1) * targets, dim=1)

    return measure, total

def _cross_entropy_compute(measures, total, reduction: Optional[str] = "mean") -> Tensor:
    """Compute the `Cross Entropy Loss`:
    WARNING: untested
    """
    if reduction == "mean":
        return measures.sum() / total
    if reduction is None or reduction == "none":
        return measures

class CrossEntropyMetric(Metric):
    is_differentiable = True
    higher_is_better = False 

    total: Tensor

    def __init__(
        self, 
        reduction: Optional[str] = "mean",
        compute_on_step: bool = True, 
        dist_sync_on_step: bool = False,
        process_group: Optional[Any] = None, 
        dist_sync_fn: Callable = None
    ) -> None:
        super().__init__(
            compute_on_step=compute_on_step, 
            dist_sync_on_step=dist_sync_on_step, 
            process_group=process_group, 
            dist_sync_fn=dist_sync_fn
        )

        allowed_reduction = ["mean", "none", None]

        if reduction not in allowed_reduction:
            raise ValueError(f"Expected argument `reduction` to be one of {allowed_reduction} but got {reduction}")
        
        self.reduction = reduction

        if self.reduction == "mean":
            self.add_state("measures", torch.zeros(1), dist_reduce_fx="sum")
        else:
            self.add_state("measure", torch.zeros(1), dist_reduce_fx="cat")

        self.add_state("total", torch.zeros(1), dist_reduce_fx="sum")

    def update(self, targets: Tensor, logits: Tensor) -> None:
        measures, total = _cross_entropy_update(targets, logits)

        if self.reduction is None or self.reduction == "none":
            self.measure.append(measures)
        else: 
            self.measures += measures.sum()
            self.total += total

    def compute(self) -> Tensor:
        measures = dim_zero_cat(self.measures) if self.reduction is None or self.reduction == "none" else self.measures
        return _cross_entropy_compute(measures, self.total, self.reduction)