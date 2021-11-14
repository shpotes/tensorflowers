from datasets import load_dataset
import pytorch_lightning as pl
from torch.utils import data
import torch
from torchvision.transforms import ToTensor
from PIL import Image

def to_one_hot_encoding(label, max_classes=20):
    one_hot = torch.zeros(max_classes)
    one_hot[label] = 1
    return one_hot.int()

class TFColDataset(data.Dataset):
    def __init__(self, split, image_transforms, target_transforms):
        self.ds = load_dataset('shpotes/tfcol')[split]

        self.image_transforms = image_transforms
        self.target_transforms = target_transforms

    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self, index):
        info_dict = self.ds[index]
        image = Image.open(info_dict['image']).convert("RGB")

        metadata = {
            'lat': info_dict['lat'],
            'lon': info_dict['lon'],
        }

        return {
            "input": self.image_transforms(image),
            "metadata": metadata,
            "target": self.target_transforms(info_dict['labels'])
        }

class TFColDataModule(pl.LightningDataModule):
    def __init__(
        self, 
        batch_size=32,
        image_transforms=ToTensor(),
        target_transforms=to_one_hot_encoding,
        features_preprocessing=lambda x: x,
        num_workers=2,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.image_transforms = image_transforms
        self.target_transforms = target_transforms
        self.features_preprocessing = features_preprocessing
        self.num_workers = num_workers

    def prepare_data(self):
        load_dataset('shpotes/tfcol')

    def setup(self, stage=None):
        self.train_dataset = TFColDataset(
            'train', 
            self.image_transforms, 
            self.target_transforms
        )

        self.val_dataset = TFColDataset(
            'validation', 
            self.image_transforms, 
            self.target_transforms
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


