import ads3
import torch
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import torchvision.datasets as datasets

from pathlib import Path
import argparse
import warnings
import random
import os

from shared.dataset import CarDataset
from shared_queues.trainer import Trainer

INPUT_SIZE = 224
data_path = Path(os.environ["DATA_PATH"])

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser()
parser.add_argument('--batch-size', type=int, default=80)
parser.add_argument('--training-workers', type=int, default=1)
parser.add_argument('--validation-workers', type=int, default=1)
parser.add_argument('--prefetch-factor', type=int, default=1)
parser.add_argument('--seed', type=int, default=1234)
parser.add_argument('--epochs', type=int, default=5)
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('--debug_data_dir', metavar='DIR', nargs='?', default='',
                    help='path to store data generated by dataloader')
parser.add_argument('--log_path', metavar='LOG_PATH', nargs='?', default='',
                    help='path to store training log')
parser.add_argument('--pretrained', action='store_true', help="use pretrained model")

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(INPUT_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

valid_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(INPUT_SIZE),
        transforms.ToTensor(),
        normalize,
    ])

if __name__ == "__main__":

    """Initialise dataset"""
    traindir = os.path.join(data_path, 'train')
    valdir = os.path.join(data_path, 'val')
    args = parser.parse_args()

    if args.seed is not None:
        print(f"Setting seed {args.seed}")
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        cudnn.benchmark = False
        print('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably!')

    pretrained = "imagenet" if args.pretrained else None

    device = torch.device("cuda")

    train_dataset = datasets.ImageFolder(
            traindir,
            train_transforms)
    
    val_dataset = datasets.ImageFolder(
            valdir,
            valid_transforms)

    model = torchvision.models.__dict__[args.arch](pretrained=True)
    model.name = args.arch + "_pretrained" if pretrained else args.arch

    print(f"PID: {os.getpid()}, Model: {model.name}")

    model_trainer = Trainer(args, model, device, train_dataset, val_dataset)

    model_trainer.train(-1)
