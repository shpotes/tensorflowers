import torch
import torch.nn as nn
import pytorch_lightning as pl

from src.dataset.pl_datamodule import DummyDataset
from src.models import HydraModule

def test_forward_pass():
    simplest_backbone = nn.Sequential(
        nn.Flatten(),
        nn.Linear(128, 128)
    )
    model = HydraModule(
        simplest_backbone,
        input_size=(1, 2, 8, 8)
    )
    model(torch.randn(1, 2, 8, 8))

def test_train_val_loop():
    dummy_data = DummyDataset((1, 2, 8, 8))
    train_dloader = torch.utils.data.DataLoader(dummy_data)
    val_dloader = torch.utils.data.DataLoader(dummy_data)

    simplest_backbone = nn.Sequential(
        nn.Flatten(),
        nn.Linear(128, 128)
    )
    model = HydraModule(
        simplest_backbone,
        input_size=(1, 2, 8, 8)
    )
    
    trainer = pl.Trainer(fast_dev_run=True)
    trainer.fit(model, train_dloader, val_dloader)