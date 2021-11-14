import torch.nn as nn
import pytorch_lightning as pl
from torchvision import models
from torchvision import transforms

from src.dataset import TFColDataModule
from src.models.pl_module import HydraModule

dm = TFColDataModule(
    image_transforms=transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
)

base_backbone = models.resnet18(pretrained=True)
backbone = nn.Sequential(*list(base_backbone.children())[:-1])
model = HydraModule(
    backbone,
    lr=3e-4,
)

trainer = pl.Trainer(overfit_batches=10)
trainer.fit(model, dm)