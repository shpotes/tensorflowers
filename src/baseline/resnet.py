import torch
import flash
from torchvision import transforms
from flash.image import ImageClassifier

from dataset import TFColDataModule

datamodule = TFColDataModule(
    batch_size=32,
    image_transforms=transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]),
)

model = ImageClassifier(backbone="resnet50", num_classes=20, multi_label=True)

trainer = flash.Trainer(max_epochs=5, gpus=torch.cuda.device_count())
trainer.finetune(
    model, 
    datamodule=datamodule,
    strategy="freeze",
)

print(trainer.validate())

trainer.save_checkpoint("weights/baseline.pt")