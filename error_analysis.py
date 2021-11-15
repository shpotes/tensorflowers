from torchvision import models
from torchvision import transforms
import pytorch_lightning as pl
import torch
from src.dataset import TFColDataModule
from src.models.pl_module import HydraModule
from sklearn.metrics import confusion_matrix
def eval():
    torch.cuda.set_per_process_memory_fraction(0.9, 0)
    torch.cuda.empty_cache()
    base_backbone = models.resnet18(pretrained=True)
    backbone = torch.nn.Sequential(*list(base_backbone.children())[:-1])

    model = HydraModule.load_from_checkpoint(
        "weights/baseline-lr_finder.ckpt",
        backbone=backbone,
    ).eval().cuda()
    dm = TFColDataModule(
            image_transforms=transforms.Compose([
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ]
        ),
            batch_size=700
    )
    dm.setup()
    for batch in dm.val_dataloader():
        input_tensor = batch['input'].cuda()
        pred = model(input_tensor)
        pred = pred['clf'].cpu().numpy()
        label = batch['target'].numpy()
    print(confusion_matrix(label, pred))

if __name__ == '__main__':
    eval()
