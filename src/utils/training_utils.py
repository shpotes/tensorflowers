import os
import requests
import torch.nn as nn
import timm

def turn_off_bn(model):
    for child in model.children():
        if isinstance(child, nn.BatchNorm2d) or isinstance(child, nn.LayerNorm):
            child.requires_grad = False
        else:
            turn_off_bn(child)

def freeze_params(backbone):
    for child in backbone.children():
        child.requires_grad = False

def load_backbone(
    model_name: str,
    pretrained: bool = False,
    checkpoint: str = "",
    weights_path: str = "weights",
):
    if not checkpoint or os.path.exists(checkpoint):
        checkpoint_path = checkpoint
    else:
        checkpoint_path = os.path.join(
            weights_path, 
            os.path.basename(checkpoint)
        )
        if not os.path.exists(checkpoint_path):
            response = requests.get(checkpoint)
            if response.status_code == 200:
                with open(checkpoint_path, "wb") as buf:
                    buf.write(response.content)
            else:
                raise RuntimeError("Download fail :c")      

    base_backbone = timm.create_model(
        model_name,
        pretrained,
        checkpoint_path=checkpoint_path
    )

    backbone = nn.Sequential(*list(base_backbone.children())[:-1])

    return backbone

