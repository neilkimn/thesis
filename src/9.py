import ads3 as ads3
#import ads3_pt_profile as ads3
import torch
import torchvision.transforms as transforms
import os
from pathlib import Path
from PIL import Image
from torch.utils import data as D

torch.manual_seed(0)
INPUT_SIZE = 224
root = Path(os.environ["DATA_PATH"])
file_train = root / "train.txt"
folder_images = root / "image"

images_train = root / "image_train"
images_valid = root / "image_valid"

class CarDataset(D.Dataset):
    def __init__(self, labels: list):
        self.filenames = []
        self.labels = labels

        """Read the dataset index file"""
        with open(file_train, newline="\n") as trainfile:
            for line in trainfile:
                self.filenames.append(folder_images / line.strip())

    def __getitem__(self, index: int):
        """Get a sample from the dataset"""
        image = Image.open(str(self.filenames[index]))
        labelStr = self.filenames[index].parts[-3]
        label = self.labels.index(labelStr)
        return image, label

    def __len__(self):
        """
        Total number of samples in the dataset
        """
        return len(self.filenames)

# In order to apply transformations specific to either training or validation data 
# I use the following class. Inspired from https://stackoverflow.com/a/59615584
class DatasetFromSubset(D.Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset)

train_transforms = transforms.Compose(
    [
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomPerspective(p=0.5),

        transforms.RandomApply(torch.nn.ModuleList([
            transforms.ColorJitter(contrast=0.5, saturation=0.5, hue=0.5),
        ]), p=0.5),

        transforms.RandomApply(torch.nn.ModuleList([
            transforms.Grayscale(num_output_channels=3),
        ]), p=0.5),

        transforms.ToTensor(),
    ]
)

valid_transforms = transforms.Compose(
    [
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.ToTensor(),
    ]
)


if __name__ == "__main__":
    """Initialise dataset"""
    labels = ads3.get_labels()
    dataset = CarDataset(labels=labels)

    num_workers = 4
    print(f"Running training script with {num_workers} workers")

    log_name = f"results/9.csv"

    """Split train and test"""
    train_len = int(0.7 * len(dataset))
    valid_len = len(dataset) - train_len
    train, valid = D.random_split(dataset, lengths=[train_len, valid_len])

    train = DatasetFromSubset(train, train_transforms)
    valid = DatasetFromSubset(valid, valid_transforms)

    # When running image augmentation you should define seperate training and validation!

    print("train size: %d, valid size %d" % (len(train), len(valid)))

    loader_train = D.DataLoader(
        train,
        batch_size=80,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=2,
    )
    loader_valid = D.DataLoader(
        valid,
        batch_size=80,
        shuffle=True,
        num_workers=1,
        pin_memory=True,
        prefetch_factor=2,
    )

    ads3.run_experiment(
        loader_train, loader_valid, log_name, 10
    )  # For profiling feel free to lower epoch count via epoch=X