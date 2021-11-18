from typing import Callable, Tuple, Optional
from datasets.features import Value

import timm.loss
import timm.scheduler
import torch
import torch.nn as nn
import pytorch_lightning as pl 

from einops import rearrange
from src.evaluation import CrossEntropyMetric
from src.loss import SparseCrossEntropyLoss, BinaryCrossEntropy
from src.data.mixup import CustomMixup

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
        clf_weight: Callable = lambda pos: 1,
        clf_loss: str = "bce",
        input_size: Tuple[int] = (1, 3, 224, 224),
        class_weight: Optional[torch.Tensor] = None,
        num_epochs: int = 35,
        warmup_lr: int = 5,
        with_mixup: bool = False,
        mixup_alpha=0.1,
        cutmix_alpha=1.0,
        mixup_prob=1.0,
        mixup_switch_prob=0.5,
        mode="batch",
        label_smoothing=0,
    ):
        super().__init__()

        self.lr = lr
        self.num_epochs = num_epochs
        self.warmup_lr = warmup_lr
        latent_size = _get_latent_size(backbone, input_size)

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
            self.clf_criterion = nn.BCEWithLogitsLoss(pos_weight=class_weight)
        if clf_loss == "asl":
            self.clf_criterion = timm.loss.AsymmetricLossMultiLabel()
        if clf_loss == "ce":
            self.clf_criterion = SparseCrossEntropyLoss()

        self.mixup_transform = lambda x, _: x
        if with_mixup:
            assert clf_loss == "bce", ValueError("Oopsy we only support mixup with BCELoss")
            self.clf_criterion = BinaryCrossEntropy(pos_weight=class_weight)
            self.city_criterion = BinaryCrossEntropy()

            self.mixup_transform = CustomMixup(
                mixup_alpha=mixup_alpha,
                cutmix_alpha=cutmix_alpha,
                prob=mixup_prob,
                switch_prob=mixup_switch_prob,
                mode=mode,
                label_smoothing=label_smoothing,
            )


        self.clf_weight = clf_weight

        self.city_head = nn.Sequential(
            nn.Linear(latent_size, 256),
            nn.ReLU(),
            nn.Linear(256, 32),
            nn.ReLU(),
            nn.Linear(32, 3),
        )

        self._optimizer_fn = lambda params: torch.optim.Adam(params, lr=lr)

    def forward(self, x):
        latent = self.feature_extraction(x)
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

        batch = self.mixup_transform(batch, self.device)

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

        lamb = self.clf_weight(batch_id)
        loss = (city_loss + lamb * clf_loss) / (lamb + 1)

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

        optimizer = torch.optim.SGD(
            trainable_params,
            lr=self.lr,
            momentum=0.9,
            weight_decay=1e-4,
        )

        decay = {
            "scheduler": timm.scheduler.CosineLRScheduler(
                optimizer,
                t_initial=self.num_epochs,
                warmup_lr_init=self.warmup_lr
            ),
            "monitor": "val_cross_entropy_loss"
        }

        return optimizer, decay


        


    
