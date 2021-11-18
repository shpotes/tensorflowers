from datasets import load_dataset
import pytorch_lightning as pl
from torch.utils import data
import torch
from torchvision.transforms import Compose, Resize, Normalize, ToTensor
from PIL import Image
from timm.data import FastCollateMixup

def to_one_hot_encoding(label, max_classes=20):
    one_hot = torch.zeros(max_classes)
    one_hot[label] = 1
    return one_hot.int()

class TFColDataset(data.Dataset):
    def __init__(self, split, image_transforms, target_transforms):
        self.ds = load_dataset('shpotes/tfcol')[split]

        self.split = split

        self.image_transforms = image_transforms
        self.target_transforms = target_transforms

    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self, index):
        info_dict = self.ds[index]
        image = Image.open(info_dict['image']).convert("RGB")

        if self.split == "test":
            return {
                "input": self.image_transforms(image)
            }

        metadata = {
            'lat': info_dict['lat'],
            'lon': info_dict['lon'],
        }
        if metadata['lat'] < 6:
            city =  0 # Bogota
        elif metadata['lat'] < 6.5:
            city = 1 # Medellín
        else:
            city = 2 # Bucaramanga

        return {
            "input": self.image_transforms(image),
            "metadata": metadata,
            "target": self.target_transforms(info_dict['labels']),
            "city": city
        }

class DummyDataset(data.Dataset):
    def __init__(self, input_shape=(1, 3, 224, 224), size=1000):
        self.input_shape = input_shape
        self.size = size
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, _):
        return {
            "input": torch.randn(*self.input_shape),
            "metadata": {"lat": 0, "lon": 0},
            "target": to_one_hot_encoding([0, 1]),
            "city": 0,
        }

class TFColDataModule(pl.LightningDataModule):
    def __init__(
        self, 
        batch_size=32,
        image_train_transforms=ToTensor(),
        target_train_transforms=to_one_hot_encoding,
        image_eval_transforms=ToTensor(),
        target_eval_transforms=to_one_hot_encoding,
        num_workers=2,
        with_mixup=False,
        mixup_alpha=0.1,
        cutmix_alpha=1.0,
        cutmix_minmax=0.5,
        prob=1.0,
        switch_prob=0.5,
        mode="batch",
        label_smoothing=0,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.image_train_transforms = image_train_transforms
        self.target_train_transforms = target_train_transforms
        self.image_eval_transforms = image_eval_transforms
        self.target_eval_transforms = target_eval_transforms
        self.num_workers = num_workers

        if with_mixup:
            self.collate_fn = FastCollateMixup(
                mixup_alpha=mixup_alpha,
                cutmix_alpha=cutmix_alpha,
                cutmix_minmax=cutmix_minmax,
                prob=prob,
                switch_prob=switch_prob,
                mode=mode,
                label_smoothing=label_smoothing,
                num_classes=20,
            )

    def prepare_data(self):
        load_dataset('shpotes/tfcol')

    def setup(self, stage=None):
        self.train_dataset = TFColDataset(
            'train', 
            self.image_train_transforms, 
            self.target_train_transforms
        )

        self.val_dataset = TFColDataset(
            'validation', 
            self.image_eval_transforms, 
            self.target_eval_transforms
        )

        self.test_dataset = TFColDataset(
            split="test",
            image_transforms=self.image_eval_transforms,
            target_transforms=None
        )

    def train_dataloader(self):
        return data.DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True,
            pin_memory=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return data.DataLoader(
            self.val_dataset, 
            batch_size=self.batch_size,
            pin_memory=True,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )
