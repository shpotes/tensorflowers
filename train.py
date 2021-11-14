import torch.nn as nn
import torch
import pytorch_lightning as pl
from torchvision import models
from torchvision import transforms
from pytorch_lightning.loggers import WandbLogger
from src.dataset import TFColDataModule
from src.models.pl_module import HydraModule

def train():
    torch.cuda.set_per_process_memory_fraction(0.8, 0)
    torch.cuda.empty_cache()

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
    base_backbone_dropout = nn.Dropout(p=0.7)(base_backbone)
    backbone = nn.Sequential(*list(base_backbone.children())[:-1])
    model = HydraModule(
        backbone,
        lr=3e-4,
    )

    logger = WandbLogger(
        project="challenge", 
        name="baseline_for_EA",
        entity="tensorflowers",
    )
    trainer = pl.Trainer(
        max_epochs=10,
        gpus=1,
        logger=logger    
    )
    trainer.fit(model, dm)
    trainer.save_checkpoint("weights/baseline.ckpt")

if __name__ == '__main__':
    train()