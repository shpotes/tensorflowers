from typing import Tuple

import torch
import torch.nn as nn
import pytorch_lightning as pl 

from einops import rearrange

def _get_latent_size(backbone: nn.Module, input_size: torch.Tensor) -> int:
    batch_size = input_size[0]
    input_tensor = torch.zeros(*input_size)
    output_tensor = backbone(input_tensor).reshape(batch_size, -1)
    
    return output_tensor.size(1)

class HydraModule(pl.LightningModule):
    def __init__(
        self,
        backbone: nn.Module,
        lr: float = 1e-3,
        input_size: Tuple[int] = (1, 3, 224, 224)
    ):
        super().__init__()

        self.lr = lr
        latent_size = _get_latent_size(backbone, input_size)
        
        self.feature_extraction = backbone
        self.classification_head = nn.Sequential(
            nn.Linear(latent_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 20),
        )

        self.clf_criterion = nn.BCEWithLogitsLoss()

        """
        self.city_head = nn.Sequential([
            nn.Linear(latent_size, 256),
            nn.ReLU(),
            nn.Linear(256, 32),
            nn.ReLU(),
            nn.Linear(32, 3),
        ])

        
        self.contrastive_head = nn.Sequential([
            nn.Linear(latent_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 20),
        ])
        """

    def forward(self, x):
        latent = self.feature_extraction(x)
        latent = rearrange(latent, "b f w h -> b (f w h)")

        clf_logits = self.classification_head(latent)
        #city_logits = self.city_head(latent)
        #dist_logitss = self.contrastive_head(latent)

        return {
            "clf": clf_logits, # b 20
            #"city": city_logits,
            #"dist_logits": dist_logitss
        }

    def _common_step(self, batch):
        """
              -> xx -> [0, 1, 0]       
        DDDDD -> xxx -> [0, 1, 0, ..., 1]
              -> xxx -> A

              -> xxx ->  B      
        DDDDD -> xxx -> [0, 1, 0, ..., 1]
              -> xx -> [0, 1, 0]

        A <-> B | C >-< A  triplet
        """
        img = batch["input"]
        clf_targets = batch["target"]
        output_dict = self.forward(img)
        clf_logits = output_dict["clf"]

        loss = self.clf_criterion(clf_logits, clf_targets.float())

        return loss

    def training_step(self, batch, _) -> float:
        train_loss = self._common_step(batch)

        self.log("train_bce_loss", train_loss)

        return train_loss

    def validation_step(self, batch, _):
        val_loss = self._common_step(batch)

        self.log("val_bce_loss", val_loss)


    def configure_optimizers(self):
        params = list(self.parameters())
        trainable_params = list(filter(lambda p: p.requires_grad, params))

        optimizer = torch.optim.Adam(trainable_params, lr=self.lr)
        return optimizer


        


    
