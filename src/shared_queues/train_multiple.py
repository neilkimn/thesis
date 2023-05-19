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
import shutil
import argparse
import random
import time
import os

from shared.dataset import CarDataset, DatasetFromSubset
from shared_queues.trainer import ProcTrainer
from shared.util import Counter, get_transformations, write_debug_indices

INPUT_SIZE = 224
data_path = Path(os.environ["DATA_PATH"])

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))
my_datasets = ["compcars", "imagenet", "imagenet64x64"]

parser = argparse.ArgumentParser()
parser.add_argument('--batch-size', type=int, default=80)
parser.add_argument('--num-processes', type=int, default=2)
parser.add_argument('--training-workers', type=int, default=1)
parser.add_argument('--validation-workers', type=int, default=1)
parser.add_argument('--prefetch-factor', type=int, default=1)
parser.add_argument('--seed', type=int, default=None)
parser.add_argument('--epochs', type=int, default=5)
parser.add_argument('--use-dali', action='store_true', help="whether to use DALI for data-loading")
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('-a','--arch', nargs='+', metavar='ARCH', help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)', default='resnet18')
parser.add_argument('--dataset', metavar='DATASET', help='dataset: ' +
                        ' | '.join(my_datasets) +
                        ' (default: compcars)', default='compcars')

parser.add_argument('--debug_data_dir', metavar='DIR', nargs='?', default='',
                    help='path to store data generated by dataloader')
parser.add_argument('--overwrite_debug_data', type=int, default=1)
parser.add_argument('--log_dir', metavar='LOG_DIR', nargs='?', default='',
                    help='path to store training log')
parser.add_argument('--pretrained', nargs='+', metavar="PRETRAIN", help="Whether to pretrain a certain model")
parser.add_argument('--dummy-data', action='store_true', help="use fake data to benchmark")
parser.add_argument('--record_first_batch_time', action='store_true', help="Don't skip measuring time spent on first batch")

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def dali_producer(qs, device, args, producer_alive):
    pid = os.getpid()
    if args.seed:
        torch.manual_seed(args.seed)
    if args.debug_data_dir:
        if args.overwrite_debug_data:
            shutil.rmtree(args.debug_data_dir)
    debug_indices_path, debug_indices_val_path = None, None

    if args.dataset == "compcars":
        images_train = data_path / args.dataset / "image_train"
        images_valid = data_path / args.dataset / "image_valid"
    if args.dataset in ("imagenet", "imagenet64x64", "imagenet_10pct"):
        images_train = data_path / args.dataset / "train"
        images_valid = data_path / args.dataset / "val"
        
    train_loader = DALIDataset(args.dataset, images_train, args.batch_size, args.training_workers, INPUT_SIZE)
    valid_loader = DALIDataset(args.dataset, images_valid, args.batch_size, args.validation_workers, INPUT_SIZE)
    
    args.train_dataset_len = len(train_loader)
    args.train_loader_len = len(train_loader)
    args.valid_dataset_len = len(valid_loader)
    args.valid_loader_len = len(valid_loader)

    for epoch in range(1, args.epochs+1):
        if args.debug_data_dir:
            debug_indices_path = Path(args.debug_data_dir) / f"epoch_{epoch}" / f"{pid}_producer_indices.txt"
            debug_indices_path.parent.mkdir(parents=True, exist_ok=True)

        for idx, data in enumerate(train_loader.dataset):
            inputs, labels = data[0]["data"], data[0]["label"]
            indices = None
            labels = labels.long()
            inputs = Variable(inputs.to(device))
            labels = Variable(labels.to(device))
            for q in qs:
                q.queue.put((idx, inputs, labels, epoch, "train", indices))
            
            #write_debug_indices(indices, debug_indices_path, args)

        # end of training for epoch, switch to eval
        if epoch > 10:
            if args.debug_data_dir:
                debug_indices_val_path = Path(args.debug_data_dir) / f"epoch_{epoch}" / f"{pid}_producer_val_indices.txt"
                debug_indices_val_path.parent.mkdir(parents=True, exist_ok=True)

            for idx, data in enumerate(valid_loader.dataset):
                inputs, labels = data[0]["data"], data[0]["label"]
                indices = None
                labels = labels.long()
                inputs = Variable(inputs.to(device))
                labels = Variable(labels.to(device))

                for q in qs:
                    q.queue.put((idx, inputs, labels, epoch, "valid", indices))

                #write_debug_indices(indices, debug_indices_val_path, args)
        
        for q in qs:
            q.queue.put((0, None, None, epoch, "end", None))
    producer_alive.wait()


def producer(loader, valid_loader, qs, device, args, producer_alive):
    pid = os.getpid()
    if args.seed:
        torch.manual_seed(args.seed)
    if args.debug_data_dir:
        if args.overwrite_debug_data:
            shutil.rmtree(args.debug_data_dir)
    debug_indices_path, debug_indices_val_path = None, None

    for epoch in range(1, args.epochs+1):
        if args.debug_data_dir:
            debug_indices_path = Path(args.debug_data_dir) / f"epoch_{epoch}" / f"{pid}_producer_indices.txt"
            debug_indices_path.parent.mkdir(parents=True, exist_ok=True)

        for idx, (inputs, labels) in enumerate(loader):
            indices = None
            inputs = Variable(inputs.to(device))
            labels = Variable(labels.to(device))

            for q in qs:
                q.queue.put((idx, inputs, labels, epoch, "train", indices))

            #write_debug_indices(indices, debug_indices_path, args)

        # end of training for epoch, switch to eval
        if epoch > 10:
            if args.debug_data_dir:
                debug_indices_val_path = Path(args.debug_data_dir) / f"epoch_{epoch}" / f"{pid}_producer_val_indices.txt"
                debug_indices_val_path.parent.mkdir(parents=True, exist_ok=True)

            for idx, (inputs, labels) in enumerate(valid_loader):
                indices = None
                inputs = Variable(inputs.to(device))
                labels = Variable(labels.to(device))

                for q in qs:
                    q.queue.put((idx, inputs, labels, epoch, "valid", indices))

                #write_debug_indices(indices, debug_indices_val_path, args)
        
        for q in qs:
            q.queue.put((0, None, None, epoch, "end", None))
    producer_alive.wait()



class MyQueue(object):
    def __init__(self, queue, index):
        self.queue = queue
        self.index = index

class Logger(object):
    def __init__(self, args, pid, log_path=None, gpu_path=None):
        self.args = args
        self.pid = pid
        self.log_path = log_path
        self.gpu_path = gpu_path
        
        self.train_time = 0
        self.batch_time = 0
        self.val_acc = 0
        self.val_loss = 0
        self.val_correct = 0
        self.val_time = 0
    
    def log_train_interval(self, idx, epoch, num_items, loss, items_processed, train_time, batch_time):
        self.train_time = train_time
        self.batch_time = batch_time

        if idx % self.args.log_interval == 0:
            print('{}\tTrain Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.2f} Throughput [img/s]: {:.1f}'.format(
                self.pid, epoch, idx * num_items, self.args.train_dataset_len,
                100. * idx / self.args.train_loader_len, loss.item(), items_processed/(train_time+batch_time)))

    def log_validation(self, val_loss, val_correct, val_acc, val_time):
        self.val_time = val_time
        self.val_acc = val_acc
        self.val_loss = val_loss
        self.val_correct = val_correct
        
    def log_write_epoch_end(self, epoch, epoch_time, train_acc, train_running_corrects):
        print('{} Validation: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            self.pid, self.val_loss, self.val_correct, self.args.valid_dataset_len, self.val_acc))
        
        print(f"{self.pid} Epoch {epoch} end: {round(epoch_time,1)}s, Train accuracy: {round(train_acc,2)}")
        if self.args.log_dir:
            with open(self.log_path, "a") as f:
                f.write(f"{int(time.time())},{epoch},{train_acc},{self.val_acc},{self.train_time},{self.batch_time},{self.val_time},{epoch_time},{train_running_corrects},{self.val_correct}\n")
                os.system(f"nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader >> {self.gpu_path}")


def worker(q, model, args, producer_alive, finished_workers):
    pid = os.getpid()
    log_path, gpu_path = None, None
    if model.on_device == False:
        print(f"{pid}\tTraining model: {model.name}")
        model.send_model()
        model.scheduler.step()
        if args.log_dir:
            log_path, gpu_path = model.init_log(pid)
    
    logger = Logger(args, pid, log_path, gpu_path)    
    if not args.record_first_batch_time:
        print("Skipping recording batch time for first batch!")
    
    debug_indices_path, debug_indices_val_path = None, None

    epochs_processed = 0

    train_time, val_time, batch_time, items_processed = 0,0,0,0
    while True:
        pid = os.getpid()
        
        start = time.time()
        idx, inputs, labels, epoch, batch_type, indices = q.queue.get()

        if args.debug_data_dir:
            debug_indices_path = Path(args.debug_data_dir) / model.name / f"{pid}_epoch_{epoch}" / "indices.txt"
            debug_indices_path.parent.mkdir(parents=True, exist_ok=True)
            debug_indices_val_path = Path(args.debug_data_dir) / model.name / f"{pid}_epoch_{epoch}" / "val_indices.txt"
            debug_indices_val_path.parent.mkdir(parents=True, exist_ok=True)

        if args.record_first_batch_time:
            batch_time += time.time() - start
        else:
            if idx > 0:
                batch_time += time.time() - start

        start = time.time()
        if batch_type == "train":
            #write_debug_indices(indices, debug_indices_path, args)
            #inputs = Variable(inputs.to(args.device))
            #labels = Variable(labels.to(args.device))

            loss = model.forward(inputs, labels)
            items_processed += len(inputs)
            train_time += time.time() - start
            logger.log_train_interval(idx, epoch, len(inputs), loss, items_processed, train_time, batch_time)

        elif batch_type == "valid":
            #write_debug_indices(indices, debug_indices_val_path, args)
            #inputs = Variable(inputs.to(args.device))
            #labels = Variable(labels.to(args.device))

            val_loss, val_acc, val_correct = model.validate(inputs, labels)
            val_time += time.time() - start
            logger.log_validation(val_loss, val_correct, val_acc, val_time)
            
        elif batch_type == "end":
            train_epoch_acc, train_running_corrects = model.end_epoch(args)
            epoch_time = train_time + val_time + batch_time
            logger.log_write_epoch_end(epoch, epoch_time, train_epoch_acc, train_running_corrects)
            train_time, val_time, batch_time,items_processed = 0,0,0,0

        q.queue.task_done()

        if batch_type == "end":
            epochs_processed += 1
            if epochs_processed == args.epochs:
                finished_workers.increment()
                if finished_workers.value() == args.num_processes:
                    producer_alive.set()
                break

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

    pretrain = ["imagenet" if p == "true" else None for p in args.pretrained]

    device = torch.device("cuda")

    if args.dataset == "imagenet64x64":
        INPUT_SIZE = 64
    
    if not args.use_dali:
        train_transforms, valid_transforms = get_transformations(args.dataset, INPUT_SIZE)

        if args.dataset in ["imagenet", "imagenet_10pct", "imagenet64x64"]:
            traindir = os.path.join(data_path / args.dataset, 'train')
            valdir = os.path.join(data_path / args.dataset, 'val')

            train_dataset = datasets.ImageFolder(
                traindir,
                train_transforms)
            
            valid_dataset = datasets.ImageFolder(
                valdir,
                valid_transforms)

        elif args.dataset == "compcars":
            labels = ads3.get_labels()
            file_train = data_path / "compcars" / "train.txt"
            folder_images = data_path / "compcars" / "image"
            dataset = CarDataset(file_path=file_train, folder_images=folder_images, labels=labels)

            train_len = int(0.7 * len(dataset))
            valid_len = len(dataset) - train_len
            train_set, valid_set = D.random_split(dataset, lengths=[train_len, valid_len], generator=torch.Generator().manual_seed(42))

            train_dataset = DatasetFromSubset(train_set, train_transforms)
            valid_dataset = DatasetFromSubset(valid_set, valid_transforms)

    train_models = []
    for idx, arch in enumerate(args.arch):
        model = torchvision.models.__dict__[arch](pretrained=pretrain[idx])
        model.name = arch + "_pretrained" if pretrain[idx] else arch
        print(f"Model: {model.name}")
        proc_model = ProcTrainer(args, model, device)
        train_models.append(proc_model)

    queues = []
    for idx in range(args.num_processes):
        q = JoinableQueue(maxsize=1)
        queue = MyQueue(q, idx)
        queues.append(queue)

    # TODO: Add ImageNet dummy data too
    if args.dummy_data:
        print("=> Dummy CompCars data is used!")
        train_dataset = datasets.FakeData(11336, (3, 224, 224), 431, train_transforms)
        val_dataset = datasets.FakeData(4680, (3, 224, 224), 431, valid_transforms)

    _start = time.time()

    producer_alive = mp.Event()
    producers = []
    if not args.use_dali:
        for i in range(1):
            train_loader = D.DataLoader(
                train_dataset,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=args.training_workers,
                pin_memory=True,
                prefetch_factor=args.prefetch_factor,
                persistent_workers=True,
                drop_last=True
            )
    
            valid_loader = D.DataLoader(
                valid_dataset,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=args.validation_workers,
                pin_memory=True,
                prefetch_factor=args.prefetch_factor,
                persistent_workers=True,
                drop_last=True
            )

            args.train_dataset_len = len(train_loader.dataset)
            #print(args.train_dataset_len)
            args.train_loader_len = len(train_loader)
            #print(args.train_loader_len)

            args.valid_dataset_len = len(valid_loader.dataset)
            #print(args.valid_dataset_len)
            args.valid_loader_len = len(valid_loader)
            #print(args.valid_loader_len)

            p = Process(target=producer, args = ((train_loader, valid_loader, queues, device, args, producer_alive)))
            #p = Process(target=producer, args = ((train_loader, valid_loader, [queues[i]], device, args, producer_alive)))
            producers.append(p)
            p.start()
    else:
        from dali_dataset import DALIDataset
        for i in range(1):
            p = Process(target=dali_producer, args = ((queues, device, args, producer_alive)))
            producers.append(p)
            p.start()

        # hardcode this for now, idc
        if args.dataset == "compcars":
            args.train_dataset_len = 11211
            args.train_loader_len = 88
            args.valid_dataset_len = 4805
            args.valid_loader_len = 38
        if "imagenet" in args.dataset:
            #args.train_dataset_len = 1281089
            args.train_dataset_len = 128116
            #args.train_loader_len = 10008
            args.train_loader_len = 1001
            args.valid_dataset_len = 49995
            args.valid_loader_len = 390

    args.device = device
    finished_workers = Counter(0)
    workers = []
    for i in range(args.num_processes):
        p = Process(target=worker, daemon=True, args=((queues[i], train_models[i], args, producer_alive, finished_workers)))
        workers.append(p)
        p.start()

    for p in workers:
        p.join()

    print(f"Completed in {time.time() - _start} seconds")