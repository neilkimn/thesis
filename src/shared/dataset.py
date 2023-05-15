from PIL import Image
from torch.utils import data as D
import torch

class CarDataset(D.Dataset):
    def __init__(self, file_path, folder_images, labels: list):
        self.filenames = []
        self.labels = labels

        """Read the dataset index file"""
        with open(file_path, newline="\n") as trainfile:
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

class DatasetFromSubset(D.Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform
        self.tensor_indices = []

    def __getitem__(self, index):
        self.tensor_indices.append(index)

        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)

        #return x, y, index
        return x,y

    def __len__(self):
        return len(self.subset)

    def get_indices(self):
        return self.tensor_indices
