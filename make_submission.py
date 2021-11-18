import argparse
from pathlib import Path

from datasets import load_dataset
import pandas as pd
import torch

from src.utils import prepare_submission

def make_submission(
    checkpoint_path,
    csv_path="submission.csv",
    backbone_name="resnet50d",
    batch_size=100,
):
    probs = prepare_submission(
        backbone_name,
        checkpoint_path,
        device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
        batch_size=batch_size
    )

    ds = load_dataset("shpotes/tfcol", split="test")
    file_names = [Path(fname).stem for fname in ds["image"]]

    df = pd.DataFrame(probs.numpy())
    df["id"] = pd.Series(file_names)
    df = df.set_index("id")

    df.to_csv(csv_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create challenge submission')
    parser.add_argument("ckpt_path", help="the path of the checkpoint with which you want to submit")
    parser.add_argument("-o", "--csv_path", help="the path where the submision csv file will be saved", default="submission.csv")
    parser.add_argument("--backbone", help="the timm-friendly backbone name", default="resnet50d")
    parser.add_argument("-b", "--batch_size", help="inference batch size (default 700)", default=100)
    
    args = parser.parse_args()

    
    make_submission(
        args.ckpt_path,
        args.csv_path,
        args.backbone,
        args.batch_size
    )