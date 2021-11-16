import torch
from torch import nn
from torchvision import models
from torchvision import transforms
import pytorch_lightning as pl

from src.evaluation.pytorch import CrossEntropyMetric, _cross_entropy_update
from src.dataset import TFColDataModule
from src.models.pl_module import HydraModule

base_backbone = models.resnet18(pretrained=True)
#backbone = nn.Sequential(*list(base_backbone.children())[:-1])
"""
model = HydraModule.load_from_checkpoint(
    "/tmp/baseline-lr_finder.ckpt",
    backbone=backbone,
).eval()

dm = TFColDataModule(
    image_transforms=transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ]),
    batch_size=678,
)
dm.setup()
"""
#with torch.no_grad():
#    full_dataset = next(iter(dm.val_dataloader()))
#    print(full_dataset["input"].shape)
    #full_logits = model()