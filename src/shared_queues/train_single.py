import time
import ads3
import torch
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.models as models
from torchvision import datasets
from dali_dataset import DALIDataset

from pathlib import Path
import argparse
import random
import os
from torch.utils import data as D
from shared.dataset import CarDataset, DatasetFromSubset
from shared_queues.trainer import Trainer, NaiveTrainer
from shared.util import get_transformations

INPUT_SIZE = 224
data_path = Path(os.environ["DATA_PATH"])

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))
my_datasets = ["compcars", "imagenet", "imagenet64x64", "imagenet128x128"]

parser = argparse.ArgumentParser()
parser.add_argument('--batch-size', type=int, default=80)
parser.add_argument('--training-workers', type=int, default=1)
parser.add_argument('--validation-workers', type=int, default=1)
parser.add_argument('--prefetch-factor', type=int, default=1)
parser.add_argument('--seed', type=int, default=None)
parser.add_argument('--epochs', type=int, default=5)
parser.add_argument('--use-dali', action='store_true', help="whether to use DALI for data-loading")
parser.add_argument('--dummy-data', action='store_true', help="use fake data to benchmark")
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('--dataset', metavar='DATASET', help='dataset: ' +
                        ' | '.join(my_datasets) +
                        ' (default: compcars)', default='compcars')
parser.add_argument('--debug_data_dir', metavar='DIR', nargs='?', default='',
                    help='path to store data generated by dataloader')
parser.add_argument('--log_path', metavar='LOG_PATH', nargs='?', default='',
                    help='path to store training log')
parser.add_argument('--pretrained', action='store_true', help="use pretrained model")



if __name__ == "__main__":

    args = parser.parse_args()
    args.seed = 1234
    if args.seed is not None:
        print(f"Setting seed {args.seed}")
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        #cudnn.deterministic = False
        #cudnn.benchmark = False
        print('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably!')

    pretrained = "imagenet" if args.pretrained else None

    #device = torch.device("cuda")
    device = "cuda:0"
    
    """Initialise dataset"""
    if args.dataset in ("imagenet64x64", "imagenet64_images"):
        INPUT_SIZE = 64
    if args.dataset == "imagenet128x128":
        INPUT_SIZE = 128
    if args.dataset == "cifar10":
        INPUT_SIZE = 32
    
    if not args.use_dali:
        train_transforms, valid_transforms = get_transformations(args.dataset, INPUT_SIZE)
    if not args.use_dali:
        if args.dataset in ["imagenet", "imagenet_10pct", "imagenet64x64", "imagenet128x128", "imagenet64_images"]:
            traindir = os.path.join(data_path / args.dataset, 'train')
            valdir = os.path.join(data_path / args.dataset, 'val')

            train_dataset = datasets.ImageFolder(
                traindir,
                train_transforms)
            
            valid_dataset = datasets.ImageFolder(
                valdir,
                valid_transforms)
    
        elif args.dataset == "compcars":
            if args.dummy_data:
                print("=> Dummy CompCars data is used!")
                train_dataset = datasets.FakeData(11336, (3, 224, 224), 431, train_transforms)
                valid_dataset = datasets.FakeData(4680, (3, 224, 224), 431, valid_transforms)
            else:
                labels = ads3.get_labels()
                file_train = data_path / "compcars" / "train.txt"
                folder_images = data_path / "compcars" / "image"
                dataset = CarDataset(file_path=file_train, folder_images=folder_images, labels=labels)

                train_len = int(0.7 * len(dataset))
                valid_len = len(dataset) - train_len
                train_set, valid_set = D.random_split(dataset, lengths=[train_len, valid_len], generator=torch.Generator().manual_seed(42))

                train_dataset = DatasetFromSubset(train_set, train_transforms)
                valid_dataset = DatasetFromSubset(valid_set, valid_transforms)
        elif args.dataset == "cifar10":
            train_dataset = torchvision.datasets.CIFAR10(
                root=data_path, train=True, download=True, transform=train_transforms)
            valid_dataset = torchvision.datasets.CIFAR10(
                root=data_path, train=False, download=True, transform=valid_transforms)
    else:
        if args.dataset == "compcars":
            images_train = data_path / "compcars" / "image_train"
            images_valid = data_path / "compcars" / "image_valid"
        elif args.dataset in ("imagenet", "imagenet_10pct", "imagenet64x64", "imagenet64_images"):
            images_train = data_path / args.dataset / "train"
            images_valid = data_path / args.dataset / "val"
        elif args.dataset == "cifar10":
            images_train = data_path / args.dataset / "train"
            images_valid = data_path / args.dataset / "val"
        print("Using DALI dataloaders!")
        train_loader = DALIDataset(args.dataset, images_train, args.batch_size, args.training_workers, INPUT_SIZE)
        valid_loader = DALIDataset(args.dataset, images_valid, args.batch_size, args.validation_workers, INPUT_SIZE)

    model = torchvision.models.__dict__[args.arch](pretrained=pretrained)
    model.name = args.arch + "_pretrained" if pretrained else args.arch
    model.to(device)
    print(f"PID: {os.getpid()}, Model: {model.name}")
    
    _start = time.time()

    if args.use_dali:
        model_trainer = Trainer(args, model, device, train_loader=train_loader, valid_loader=valid_loader)
    else:
        model_trainer = Trainer(args, model, device, train_dataset=train_dataset, val_dataset=valid_dataset)

    model_trainer.train(-1)

    print(f"Completed in {time.time() - _start} seconds")