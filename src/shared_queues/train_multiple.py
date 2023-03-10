import ads3 as ads3
import torch
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import torchvision.datasets as datasets

import torch.multiprocessing as mp
from torch.multiprocessing import Process, Queue, JoinableQueue
from torch.utils import data as D
from torch.autograd import Variable

from pathlib import Path
import argparse
import warnings
import random
import time
import os

from shared.dataset import CarDataset, SharedDataset, DatasetFromSubset
from shared_queues.trainer import ProcTrainer

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
parser.add_argument('--num-processes', type=int, default=2)
parser.add_argument('--training-workers', type=int, default=1)
parser.add_argument('--validation-workers', type=int, default=1)
parser.add_argument('--prefetch-factor', type=int, default=1)
parser.add_argument('--seed', type=int, default=None)
parser.add_argument('--epochs', type=int, default=5)
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('-a','--arch', nargs='+', metavar='ARCH', help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)', default='resnet18')

parser.add_argument('--debug_data_dir', metavar='DIR', nargs='?', default='',
                    help='path to store data generated by dataloader')
parser.add_argument('--log_path', metavar='LOG_PATH', nargs='?', default='',
                    help='path to store training log')
parser.add_argument('--pretrained', nargs='+', metavar="PRETRAIN", help="Whether to pretrain a certain model")
parser.add_argument('--dummy_data', action='store_true', help="use fake data to benchmark")

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

class MyQueue(object):
    def __init__(self, queue, qi):
        self.queue = queue
        self.qi = qi

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def producer(loader, valid_loader, qs, device, args):
    pid = os.getpid()
    #print(f"Producer pid: {pid}")
    torch.manual_seed(args.seed)

    for epoch in range(1, args.epochs+1):
        torch.cuda.nvtx.range_push(f"Producer: Start epoch {epoch}")
        if args.debug_data_dir:
            debug_indices = Path(args.debug_data_dir) / f"epoch_{epoch}" / "indices.txt"
            debug_indices.parent.mkdir(parents=True, exist_ok=True)

        torch.cuda.nvtx.range_push("Start training data loading")
        for idx, (inputs, labels, indices) in enumerate(loader):
            torch.cuda.nvtx.range_push("Sending data to GPU")
            inputs = Variable(inputs.to(device))
            labels = Variable(labels.to(device))
            torch.cuda.nvtx.range_pop()
            for qi, q in enumerate(qs):
                #print(f"Putting indices {indices[:2]} in queue {qi}")
                torch.cuda.nvtx.range_push("Put batch on queue")
                q.put((idx, inputs, labels, epoch, "train", indices))
                torch.cuda.nvtx.range_pop()

            if args.debug_data_dir:
                with open(debug_indices, "a") as f:
                    f.write(" ".join(list(map(str, indices.tolist()))))
                    f.write("\n")
        torch.cuda.nvtx.range_pop()

        # end of training for epoch, switch to eval
        if args.debug_data_dir:
            debug_indices = Path(args.debug_data_dir) / f"epoch_{epoch}" / "val_indices.txt"
            debug_indices.parent.mkdir(parents=True, exist_ok=True)

        torch.cuda.nvtx.range_push("Start validation data loading")
        torch.cuda.nvtx.range_push("Getting next batch from validation loader")
        for idx, (inputs, labels, indices) in enumerate(valid_loader):
            torch.cuda.nvtx.range_pop()
            torch.cuda.nvtx.range_push("Sending data to GPU")
            inputs = Variable(inputs.to(device))
            labels = Variable(labels.to(device))
            torch.cuda.nvtx.range_pop()
            for q in qs:
                torch.cuda.nvtx.range_push("Put batch on queue")
                q.put((idx, inputs, labels, epoch, "valid", indices))
                torch.cuda.nvtx.range_pop()

            if args.debug_data_dir:
                with open(debug_indices, "a") as f:
                    f.write(" ".join(list(map(str, indices.tolist()))))
                    f.write("\n")
            torch.cuda.nvtx.range_push("Getting next batch from validation loader")

        torch.cuda.nvtx.range_pop()
        torch.cuda.nvtx.range_pop()
        torch.cuda.nvtx.range_pop()
        
        for q in qs:
            q.put((None, None, None, epoch, "end", None))
        
        print(f"Epoch {epoch} done")

        ### do eval

    pid = os.getpid()
    #print(f'producer {pid} done')

class TimeLogger(object):
    def __init__(self):
        self.training_time = 0.0
        self.validation_time = 0.0
        self.train_timer = 0.0
        self.val_timer = 0.0
        
        self.training = False
        self.validating = False

    def start_training(self):
        if not self.training:
            self.train_timer = time.time()
            self.training = True

    def start_validation(self):
        if not self.validating:
            self.val_timer = time.time()
            self.validating = True

    def stop_training(self):
        if self.training:
            self.training_time += time.time() - self.train_timer
            self.train_timer = 0.0
            self.training = False

    def stop_validation(self):
        if self.validating:
            self.validation_time += time.time() - self.val_timer
            self.val_timer = 0.0
            self.validating = False

    def reset(self):
        self.stop_training()
        self.stop_validation()
        self.training_time = 0.0
        self.validation_time = 0.0
        self.train_timer = 0.0
        self.val_timer = 0.0

    def get_row(self, num_elems):
        return self.training_time, self.validation_time, self.training_time+self.validation_time, num_elems / self.training_time 


def worker(q, model, args):
    if model.on_device == False:
        pid = os.getpid()
        print(f"{pid}\tTraining model: {model.name}")
        model.send_model()
        model.scheduler.step()
        if args.log_path:
            model.init_log(pid)

    timer = TimeLogger()
    currently_training = False
    currently_validating = False
    
    while True:
        pid = os.getpid()

        torch.cuda.nvtx.range_push("Fetched batch from queue")
        batch_time = time.time()
        idx, inputs, labels, epoch, batch_type, indices = q.get()
        batch_time = time.time() - batch_time
        torch.cuda.nvtx.range_pop()

        if args.debug_data_dir:
            debug_indices = Path(args.debug_data_dir) / model.name / f"epoch_{epoch}" / "indices.txt"
            debug_indices.parent.mkdir(parents=True, exist_ok=True)

        if args.debug_data_dir:
            val_debug_indices = Path(args.debug_data_dir) / model.name / f"epoch_{epoch}" / "val_indices.txt"
            val_debug_indices.parent.mkdir(parents=True, exist_ok=True)
        
        if batch_type == "train":
            if not currently_training:
                if currently_validating:
                    torch.cuda.nvtx.range_pop()
                    currently_validating = False
                    timer.stop_validation()
                    print(f"Subtracting {batch_time}s from val time")
                    timer.validation_time -= batch_time
                torch.cuda.nvtx.range_push(f"Child: Start train {epoch}")
                timer.start_training()
                print(f"Adding {batch_time}s to train time")
                timer.training_time += batch_time
                currently_training = True
            
            torch.cuda.nvtx.range_push(f"Child: Model forward")
            model.forward(inputs, labels, idx, epoch, pid)
            torch.cuda.nvtx.range_pop()

            if args.debug_data_dir:
                with open(debug_indices, "a") as f:
                    f.write(" ".join(list(map(str, indices.tolist()))))
                    f.write("\n")

        elif batch_type == "valid":
            if not currently_validating:
                if currently_training:
                    timer.stop_training()
                    print(f"Subtracting {batch_time}s from train time")
                    timer.training_time -= batch_time

                    torch.cuda.nvtx.range_pop()
                    currently_training = False
                torch.cuda.nvtx.range_push(f"Child: Start val {epoch}")
                timer.start_validation()
                print(f"Adding {batch_time}s to val time")
                timer.validation_time += batch_time
                currently_validating = True

            torch.cuda.nvtx.range_push(f"Child: Model validate")
            model.validate(inputs, labels)
            torch.cuda.nvtx.range_pop()
            if args.debug_data_dir:
                with open(val_debug_indices, "a") as f:
                    f.write(" ".join(list(map(str, indices.tolist()))))
                    f.write("\n")

        elif batch_type == "end":
            torch.cuda.nvtx.range_pop()
            timer.stop_training()
            timer.stop_validation()
            print(f"Training time: {timer.training_time}s, validation time {timer.validation_time}s")
            model.end_epoch(args, epoch, timer)
            timer.reset()
            
        #print(f'pid {pid} Finished {idx}')
        q.task_done()

if __name__ == "__main__":

    args = parser.parse_args()
    assert args.num_processes == len(args.arch)

    if args.seed is not None:
        print(f"Setting seed {args.seed}")
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        cudnn.benchmark = False
        print('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably!')
        
    mp.set_start_method("spawn", force=True)
    manager = mp.Manager()

    """Initialise dataset"""
    labels = ads3.get_labels()

    pretrain = ["imagenet" if p == "true" else None for p in args.pretrained]

    device = torch.device("cuda")

    dataset = CarDataset(file_path=file_train, folder_images=folder_images, labels=labels)

    train_models = []
    for idx, arch in enumerate(args.arch):
        model = torchvision.models.__dict__[arch](pretrained=pretrain[idx])
        model.name = arch + "_pretrained" if pretrain[idx] else arch
        print(f"Model: {model.name}")
        proc_model = ProcTrainer(args, model, device)
        train_models.append(proc_model)
    
    queues = [JoinableQueue(maxsize=10)]*args.num_processes

    """Split train and test"""
    train_len = int(0.7 * len(dataset))
    valid_len = len(dataset) - train_len
    train_set, valid_set = D.random_split(dataset, lengths=[train_len, valid_len], generator=torch.Generator().manual_seed(42))

    train_set = DatasetFromSubset(train_set, train_transforms)
    valid_set = DatasetFromSubset(valid_set, valid_transforms)

    if args.dummy_data:
        print("=> Dummy CompCars data is used!")
        train_dataset = datasets.FakeData(11336, (3, 224, 224), 431, train_transforms)
        val_dataset = datasets.FakeData(4680, (3, 224, 224), 431, valid_transforms)

    train_loader = D.DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.training_workers,
        pin_memory=True,
        prefetch_factor=args.prefetch_factor,
    )

    valid_loader = D.DataLoader(
        valid_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.validation_workers,
        pin_memory=True,
        prefetch_factor=args.prefetch_factor,
    )

    args.train_dataset_len = len(train_loader.dataset)
    args.train_loader_len = len(train_loader)

    args.valid_dataset_len = len(valid_loader.dataset)
    args.valid_loader_len = len(valid_loader)

    _start = time.time()
    for i in range(args.num_processes):
        args.qi = i
        p = Process(target=worker, daemon=True, args=((queues[i], train_models[i], args))).start()

    producers = []
    for i in range(1):
        p = Process(target=producer, args = ((train_loader, valid_loader, queues, device, args)))
        producers.append(p)
        p.start()

    for p in producers:
        p.join()

    for q in queues:
        q.join()
    print(f"Completed in {time.time() - _start} seconds")