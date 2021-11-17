from typing import Dict, Union

from torchvision import transforms as T
from timm.data import rand_augment_transform, RandomResizedCropAndInterpolation
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

def _rand_aug_dict_to_config_str(hparams: Dict[str, Union[int, str]]) -> str:
    if "inc" in hparams and hparams["inc"]:
        increasing = True
        del hparams["inc"]
    else: 
        increasing = False

    config_str = "rand-"
    for k, v in hparams.items():
        config_str += "{k}{v}-"
    
    if increasing:
        config_str += "inc"
    else:
        config_str = config_str[:-1] # remove extra `-`
    
    return config_str

def create_train_transformations(
    with_rand_augmentation=False,
    rand_aug_params = {
        "m": 5,
        "n": 2,
        "mstd": 0.5,
        "inc": True
    }
):
    image_preprocessing = [
        T.Resize(256),
        RandomResizedCropAndInterpolation(
            size=224,
            scale=(0.08, 1.0),  # default imagenet scale range 
            ratio=(3./4., 4./3.), # default imagenet ratio range
            interpolation="random"
        ),
        T.RandomHorizontalFlip(p=0.5)
    ]

    if with_rand_augmentation:
        aa_params = dict(
            translate_const=int(224 * 0.45),
            img_mean=tuple([min(255, round(255 * x)) for x in IMAGENET_DEFAULT_MEAN]),
        )
        
        config_str = _rand_aug_dict_to_config_str(rand_aug_params)

        train_augmentations = [
            rand_augment_transform(config_str, aa_params)
        ]
    else:
        train_augmentations = [] # Can be extended
    
    image_post_processing = [
        T.ToTensor(),
        T.Normalize(
            mean=IMAGENET_DEFAULT_MEAN,
            std=IMAGENET_DEFAULT_STD,
        ),
    ]

    return T.Compose(
        image_preprocessing + train_augmentations + image_post_processing
    )

    
