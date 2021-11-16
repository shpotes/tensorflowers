import torch.nn as nn

def turn_off_bn(model):
 for child in model.children():
    if isinstance(child, nn.BatchNorm2d) or isinstance(child, nn.LayerNorm):
     child.requires_grad = False
    else:
      turn_off_bn(child)
