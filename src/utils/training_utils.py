import os
import requests
import torch.nn as nn
import timm

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
    