from typing import Callable, Tuple, Optional

import timm.loss
import torch
import torch.nn as nn
import pytorch_lightning as pl 

from einops import rearrange
from src.evaluation import CrossEntropyMetric
from src.loss.cross_entropy import SparseCrossEntropyLoss

def _get_latent_size(backbone: nn.Module, input_size: torch.Tensor, pool_latent: bool) -> int:
    batch_size = input_size[0]
    input_tensor = torch.zeros(*input_size)
    output_tensor = backbone(input_tensor)
    
    if not pool_latent:
        output_tensor = output_tensor.reshape(batch_size, -1)
    else:
        output_tensor = output_tensor.permute(0, 2, 1) # An small hack to get the correct shape
    
    return output_tensor.size(1)

class HydraModule(pl.LightningModule):
    def __init__(
        self,
        backbone: nn.Module,
        lr: float = 1e-3,
        clf_weight: Callable = lambda pos: 1,
        clf_loss: str = "bce",
        input_size: Tuple[int] = (1, 3, 224, 224),
        class_weight: Optional[torch.Tensor] = None,
        pool_latent: bool = False,
    ):
        super().__init__()

        self.lr = lr
        self.pool_latent = pool_latent
        latent_size = _get_latent_size(backbone, input_size, self.pool_latent)

        self.train_metric = CrossEntropyMetric()
        self.val_metric = CrossEntropyMetric()
        
        self.feature_extraction = backbone
        self.classification_head = nn.Sequential(
            nn.Linear(latent_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 20),
        )

        self.city_criterion = nn.CrossEntropyLoss()

        if clf_loss == "bce":
            self.clf_criterion = nn.BCEWithLogitsLoss(weight=class_weight)
        if clf_loss == "asl":
            self.clf_criterion = timm.loss.AsymmetricLossMultiLabel()
        if clf_loss == "ce":
            self.clf_criterion = SparseCrossEntropyLoss()

        self.clf_weight = clf_weight

        self.city_head = nn.Sequential(
            nn.Linear(latent_size, 256),
            nn.ReLU(),
            nn.Linear(256, 32),
            nn.ReLU(),
            nn.Linear(32, 3),
        )

    def forward(self, x):
        latent = self.feature_extraction(x)
        if self.pool_latent:
            latent = torch.mean(latent, dim=1)
        else:
            latent = rearrange(latent, "b ... -> b (...)")

        clf_logits = self.classification_head(latent)
        city_logits = self.city_head(latent)
        
        return {
            "clf": clf_logits,
            "city": city_logits,
        }

    def _common_step(self, batch):
        """
              -> xx -> [0, 1, 0]       
        DDDDD 
              -> xxx -> [0, 1, 0, ..., 1]
        """
        img = batch["input"]
        clf_targets = batch["target"]
        city_targets = batch["city"]

        forward_dict = self(img)
        clf_logits = forward_dict["clf"]
        city_logits = forward_dict["city"]

        clf_loss = self.clf_criterion(clf_logits, clf_targets.float())
        city_loss = self.city_criterion(city_logits, city_targets.long())

        output_dict = {
            "city_loss": city_loss,
            "clf_loss": clf_loss,
            "clf_logits": clf_logits,
        }

        return output_dict

    def training_step(self, batch, batch_id) -> float:
        output_dict = self._common_step(batch)

        city_loss = output_dict["city_loss"]
        clf_loss = output_dict["clf_loss"]
        lamb = self.clf_weight(batch_id)
        loss = (city_loss + lamb * clf_loss) / (lamb + 1)

        train_cross_entropy = self.train_metric(
            output_dict["clf_logits"], 
            batch["target"].float()
        )

        self.log("train_cross_entropy_loss", train_cross_entropy, prog_bar=True)
        self.log(
            "losses", 
            {
                "train_city_loss": city_loss,
                "train_clf_loss": clf_loss,
                "train_loss": loss
            }
        )

        return loss

    def validation_step(self, batch, batch_id):
        output_dir = self._common_step(batch)

        city_loss = output_dir["city_loss"]
        clf_loss = output_dir["clf_loss"]
        loss = city_loss + self.clf_weight(batch_id) * clf_loss

        val_cross_entropy = self.val_metric(output_dir["clf_logits"], batch["target"].float())

        self.log("val_cross_entropy_loss", val_cross_entropy)
        self.log(
            "losses", 
            {
                "val_city_loss": city_loss,
                "val_clf_loss": clf_loss,
                "val_loss": loss
            }
        )

    def configure_optimizers(self):
        params = list(self.parameters())
        trainable_params = list(filter(lambda p: p.requires_grad, params))

        optimizer = torch.optim.Adam(trainable_params, lr=self.lr)
        return optimizer


        


    
