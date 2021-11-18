import timm
import torch.nn as nn
import torch
import pytorch_lightning as pl
from torchvision import models
from torchvision import transforms
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from src.dataset import TFColDataModule
from src.model import HydraModule
from src.utils.training_utils import turn_off_bn

def train(checkpoint_dir):
    dm = TFColDataModule(
        image_transforms=transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224), # this may increase label noise
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    )

    base_backbone = timm.create_model(
        model_name="resnet50d",
        checkpoint_path="weights/resnet50d_a1_0-e20cff14.pth",
    )

    backbone = nn.Sequential(*list(base_backbone.children())[:-1])
    turn_off_bn(backbone)

    model = HydraModule(
        backbone,
        lr=3e-4,
        clf_loss="asl"
    )

    logger = WandbLogger(
        project="challenge", 
        name="MH-resnet50d-timm-asl-aug",
        entity="tensorflowers",
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="val_cross_entropy_loss",
        dirpath=checkpoint_dir,
        filename="MH-resnet50d-timm-asl-aug-{epoch:02d}-{val_cross_entropy_loss:.2f}",
        save_top_k=3,
        mode="min",
    )

    trainer = pl.Trainer(
        gpus=1,
        logger=logger,
        callbacks=[checkpoint_callback]
    )
    trainer.fit(model, dm)

if __name__ == '__main__':
    train("weights/models")
