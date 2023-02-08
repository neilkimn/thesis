import ads3 as ads3
import torch
import torchvision
import torchvision.transforms as transforms

#import torch.multiprocessing as mp
#import multiprocessing as mp
import multiprocess as mp

from pathlib import Path
import numpy as np
import cupy as cp
import argparse
import os

from dataset import CarDataset, SharedDataset
from trainer import Trainer

torch.manual_seed(0)
INPUT_SIZE = 224
root = Path(os.environ["DATA_PATH"])
file_train = root / "train.txt"
folder_images = root / "image"
images_train = root / "image_train"
images_valid = root / "image_valid"

parser = argparse.ArgumentParser()
parser.add_argument('--num-processes', type=int, default=2)
parser.add_argument('--batch-size', type=int, default=80)
parser.add_argument('--num-workers', type=int, default=1)
parser.add_argument('--seed', type=int, default=1234)
parser.add_argument('--epochs', type=int, default=1)
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')

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

def start(shared_dataset):
    shared_dataset.start()

if __name__ == "__main__":

    """Initialise dataset"""
    labels = ads3.get_labels()
    args = parser.parse_args()

    device = torch.device("cuda")

    dataset = CarDataset(file_path=file_train, folder_images=folder_images, labels=labels)

    model = torchvision.models.resnet18(pretrained="imagenet")
    mp.set_start_method("spawn", force=True)
    manager = mp.Manager()
    lock = mp.Lock()
    #shared_tensors = manager.list()
    #shared_accesses = manager.list()
    shared_tensors = list()
    shared_accesses = list()

    num_batches = int(len(dataset) / args.batch_size)

    shared_dataset = SharedDataset(shared_tensors, shared_accesses, num_batches, args.num_processes, device)
    model_trainer = Trainer(args, manager, model, device, dataset, train_transforms, valid_transforms, shared_dataset)

    processes = []
    for rank in range(args.num_processes):
        p = mp.Process(target = model_trainer.train, args=(rank, lock))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()

    model_trainer.test()
