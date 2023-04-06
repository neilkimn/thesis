import ads3
import torch
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models

from pathlib import Path
import argparse
import warnings
import random
import os

from shared.dataset import CarDataset
from shared_queues.trainer import Trainer

INPUT_SIZE = 224
root = Path(os.environ["DATA_PATH"])
file_train = root / "train.txt"
folder_images = root / "image"
images_train = root / "image_train"
images_valid = root / "image_valid"

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

    dataset = CarDataset(file_path=file_train, folder_images=folder_images, labels=labels)

    model = torchvision.models.__dict__[args.arch](pretrained=pretrained)
    model.name = args.arch + "_pretrained" if pretrained else args.arch

    print(f"PID: {os.getpid()}, Model: {model.name}")

    model_trainer = Trainer(args, model, device, dataset, train_transforms, valid_transforms)

    model_trainer.train(-1)
