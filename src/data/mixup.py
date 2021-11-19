import torch
from timm.data.mixup import Mixup

def one_hot(x, num_classes, on_value=1., off_value=0., device=None):
    x = x.long().view(-1, 1)
    if x.size(1) == num_classes:
        out = torch.where(x == 1, on_value, off_value)
    else:  
        out = torch.full((x.size()[0], num_classes), off_value, device=device).scatter_(1, x, on_value)
    return out

def mixup_target(target, num_classes, lam=1., smoothing=0.0, device=None):
    off_value = smoothing / num_classes
    on_value = 1. - smoothing + off_value
    y1 = one_hot(target, num_classes, on_value=on_value, off_value=off_value, device=device)
    y2 = one_hot(target.flip(0), num_classes, on_value=on_value, off_value=off_value, device=device)
    return y1 * lam + y2 * (1. - lam)

class CustomMixup(Mixup):
    def __init__(self, mixup_alpha=1., cutmix_alpha=0., cutmix_minmax=None, prob=1.0, switch_prob=0.5,
                 mode='batch', correct_lam=True, label_smoothing=0.1):
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.cutmix_minmax = cutmix_minmax
        if self.cutmix_minmax is not None:
            assert len(self.cutmix_minmax) == 2
            # force cutmix alpha == 1.0 when minmax active to keep logic simple & safe
            self.cutmix_alpha = 1.0
        self.mix_prob = prob
        self.switch_prob = switch_prob
        self.label_smoothing = label_smoothing
        self.mode = mode
        self.correct_lam = correct_lam  # correct lambda based on clipped area for cutmix
        self.mixup_enabled = True  # set to false to disable mixing (intended tp be set by train loop)

    def __call__(self, batch, device):
        input_tensor = batch["input"]
        clf_target = batch["target"]
        city_target = batch["city"]

        assert len(input_tensor) % 2 == 0, 'Batch size should be even when using this'
        if self.mode == 'elem':
            lam = self._mix_elem(input_tensor)
        elif self.mode == 'pair':
            lam = self._mix_pair(input_tensor)
        else:
            lam = self._mix_batch(input_tensor)

        clf_mixup_target = mixup_target(clf_target, 20, lam, self.label_smoothing, device)
        city_mixup_target = mixup_target(city_target, 3, lam, self.label_smoothing, device)

        return {
            "input": input_tensor, 
            "target": clf_mixup_target,
            "city": city_mixup_target,
        }
