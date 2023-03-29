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

class SharedDataset:
    def __init__(self, tensors, accesses, epochs, max_accesses, device, shared_size=100):
        self.tensors = tensors
        self.accesses = accesses
        self.tensors.extend([None] * shared_size)
        self.accesses.extend([0] * shared_size)
        self.max_accesses = max_accesses
        self.device = device
        self.shared_size = shared_size

    def get_batch(self, lock, idx, pid):
        lock.acquire()
        print(f"{pid} Getting batch at idx: {idx % self.shared_size}")
        self.accesses[idx % self.shared_size] += 1
        inputs, labels = self.tensors[idx % self.shared_size]
        lock.release()
        return torch.as_tensor(inputs, device=self.device), torch.as_tensor(labels, device=self.device)

    def set_batch(self, lock, inputs, labels, idx, pid):
        lock.acquire()
        print(f"{pid} Setting batch at idx: {idx % self.shared_size}")
        self.accesses[idx % self.shared_size] += 1
        self.tensors[idx % self.shared_size] = (cp.asarray(inputs), cp.asarray(labels))
        lock.release()

    def remove_batch(self, lock, idx, pid):
        lock.acquire()
        if self.accesses[idx % self.shared_size] == self.max_accesses:
            print(f"{pid} Removing batch at idx: {idx % self.shared_size}")
            self.tensors[idx % self.shared_size] = None
            self.accesses[idx % self.shared_size] = 0
        lock.release()

    def get_accesses(self, lock, idx):
        lock.acquire()
        accesses = self.accesses[idx % self.shared_size]
        lock.release()
        return accesses
    
    def remove_all_batches(self, lock):
        lock.acquire()
        for idx in range(self.shared_size):
            self.tensors[idx % self.shared_size] = None
            self.accesses[idx % self.shared_size] = 0
        lock.release()
        

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

        return x, y, index

    def __len__(self):
        return len(self.subset)

    def get_indices(self):
        return self.tensor_indices
