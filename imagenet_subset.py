import os
from pathlib import Path
import shutil

DATA_PATH = Path(os.environ["DATA_PATH"])

def main():
    imagenet_path = DATA_PATH / "imagenet"
    imagenet_train_path = imagenet_path / "train"
    imagenet_val_path = imagenet_path / "val"
    imagenet_subset_path = DATA_PATH / "imagenet_10pct"
    imagenet_subset_train_path = imagenet_subset_path / "train"
    imagenet_subset_val_path = imagenet_subset_path / "val"
    os.makedirs(imagenet_subset_train_path, exist_ok=True)
    os.makedirs(imagenet_subset_val_path, exist_ok=True)

    with open("imagenet_10pct.txt") as f:
        lines = f.readlines()
        files = [l.strip() for l in lines]
    for file_name in files:
        sub_dir = file_name.split("_")[0]
        file_path = imagenet_train_path / sub_dir / file_name
        dest_dir = imagenet_subset_train_path / sub_dir
        os.makedirs(dest_dir, exist_ok=True)
        dest_path = dest_dir / file_name

        shutil.copyfile(file_path, dest_path)

    shutil.copytree(imagenet_val_path, imagenet_subset_val_path, dirs_exist_ok=True)

if __name__ == '__main__':
    main()