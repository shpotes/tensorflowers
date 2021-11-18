from typing import Any, Dict, Union, Optional, Tuple

import torch
import torch.nn.functional as F
import datasets
import timm
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from src.dataset import TFColDataModule
from src.model import HydraModule

_INFO = datasets.load_dataset_builder("shpotes/tfcol").info
_CLASS_NAMES = _INFO.features["labels"].feature.names
_NAMES_MAP = dict(zip(_CLASS_NAMES, range(20)))

def int2str(idx):
  return _CLASS_NAMES[idx]

def str2int(label):
    return _NAMES_MAP[label]

def load_model_for_inference(
  backbone: Union[str, nn.Module], 
  ckpt_path: str
) -> nn.Module:
  if isinstance(backbone, str):
    base_backbone = timm.create_model(backbone)
    backbone = backbone = nn.Sequential(*list(base_backbone.children())[:-1])

  module = HydraModule.load_from_checkpoint(
    ckpt_path,
    backbone=backbone,
  )

  model = nn.Sequential(
    *module.feature_extraction.children(),
    nn.Flatten(),
    *module.classification_head.children(),
  ).eval()

  return model

def memory_contraint_full_inference(
  model: nn.Module,
  dloader: DataLoader,
  with_targets: bool = False,
  device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:  
  if torch.cuda.is_available():
    WITH_CUDA = True

  output_dict = {
    "logits": [],
  }      
  if with_targets:
    output_dict["targets"] = []

  with torch.no_grad():
    for batch in dloader:
      cpu_tensor = batch["input"]

      for layer in model.children():
        layer.cuda()

        gpu_tensor = cpu_tensor.cuda()
        gpu_tensor = layer(gpu_tensor)
        cpu_tensor = gpu_tensor.cpu()

        layer.cpu()
        
        del gpu_tensor
        torch.cuda.empty_cache()

      output_dict["logits"].append(cpu_tensor)
      if with_targets:
        output_dict["target"].append(batch["target"])
    
  
  return {k: torch.cat(v) for k, v in output_dict.items()}

def prepare_submission(
  backbone: Union[str, nn.Module],
  ckpt_path: str,
  device: Optional[torch.device] = None,
  image_transforms =  transforms.Compose([
      transforms.Resize(224),
      transforms.ToTensor(),
      transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
      )
    ]),
  batch_size=700,
) -> torch.Tensor:
  model = load_model_for_inference(backbone, ckpt_path)
  dm = TFColDataModule(batch_size=batch_size, image_transforms=image_transforms)
  dm.setup()

  output_dict = memory_contraint_full_inference(
    model, 
    dm.test_dataloader(),
    with_targets=False,
    device=device
  )

  logits = output_dict["logits"]
  probs = logits.softmax(axis=1)

  return probs