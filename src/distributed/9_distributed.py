import argparse
import os
import random
import time
import warnings
from enum import Enum

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Subset
from torch.autograd import Variable

#from src.utils import Summary, AverageMeter, ProgressMeter, accuracy, save_checkpoint
class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f', summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def all_reduce(self):
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        total = torch.tensor([self.sum, self.count], dtype=torch.float32, device=device)
        dist.all_reduce(total, dist.ReduceOp.SUM, async_op=False)
        self.sum, self.count = total.tolist()
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
    
    def summary(self):
        fmtstr = ''
        if self.summary_type is Summary.NONE:
            fmtstr = ''
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = '{name} {avg:.3f}'
        elif self.summary_type is Summary.SUM:
            fmtstr = '{name} {sum:.3f}'
        elif self.summary_type is Summary.COUNT:
            fmtstr = '{name} {count:.3f}'
        else:
            raise ValueError('invalid summary type %r' % self.summary_type)
        
        return fmtstr.format(**self.__dict__)

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR', nargs='?', default='imagenet',
                    help='path to dataset (default: imagenet)')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=10, type=int, metavar='N',
                    help='number of total epochs to run')

parser.add_argument('-b', '--batch-size', default=80, type=int,
                    metavar='N',
                    help='batch size (default: 80)')
parser.add_argument('-p', '--print-freq', default=100, type=int,
                    metavar='N', help='print frequency (default: 100)')

parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://127.0.0.1:12355', type=str,
                    help='url used to set up distributed training')

parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
parser.add_argument('--dummy', action='store_true', help="use fake data to benchmark")
parser.add_argument('--prof', action='store_true', help="Whether to profile with pytorch profiler")
parser.add_argument('--no-memcpy', action='store_true', help="Whether to include memcpy operations")

parser.add_argument('--name', default='', type=str)

best_acc1 = 0


def main():
    args = parser.parse_args()

    #os.environ['MASTER_ADDR'] = 'localhost'
    #os.environ['MASTER_PORT'] = '12355'

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        cudnn.benchmark = False
        print(f"Setting seed: {args.seed} for deterministic behavior")

    if args.gpu is not None:
        print("Running training on single GPU")

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    if torch.cuda.is_available():
        gpus_per_node = torch.cuda.device_count()
    else:
        gpus_per_node = 1
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        # world_size adjusted wrt gpus_per_node
        args.world_size = gpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=gpus_per_node, args=(gpus_per_node, args))
    else:
        main_worker(args.gpu, gpus_per_node, args)


def main_worker(gpu, gpus_per_node, args):
    global best_acc1
    args.gpu = gpu

    if args.name:
        log_name = args.name + "_gpu" + str(args.gpu) + ".csv"
        log_file = open(log_name, "w")
        log_file.write("epoch,train_acc,valid_acc,time,data_time,train_corr,valid_corr,throughput\n")

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * gpus_per_node + gpu
        #print(f"Use GPU: {args.gpu} for training, rank {args.rank}, world size {args.world_size}")
        dist.init_process_group(backend="nccl", init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    # create model
    print(f"Pretrained model: {args.arch}")
    model = torchvision.models.__dict__[args.arch](pretrained="imagenet")
    
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, 431)

    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if torch.cuda.is_available():
            if args.gpu is not None:
                torch.cuda.set_device(args.gpu)
                model.cuda(args.gpu)
                # When using a single GPU per process and per
                # DistributedDataParallel, we need to divide the batch size
                # ourselves based on the total number of GPUs of the current node.
                args.batch_size = int(args.batch_size / gpus_per_node)
                print(f"Working on a batch size of: {args.batch_size}")
                args.workers = int((args.workers + gpus_per_node - 1) / gpus_per_node)
                model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
            else:
                model.cuda()
                # DistributedDataParallel will divide and allocate batch_size to all
                # available GPUs if device_ids are not set
                model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None and torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        model = model.to(device)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    if torch.cuda.is_available():
        if args.gpu:
            device = torch.device('cuda:{}'.format(args.gpu))
        else:
            device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")


    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    scheduler = StepLR(optimizer, step_size=7, gamma=0.1)

    train_transforms = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomPerspective(p=0.5),

                transforms.RandomApply(torch.nn.ModuleList([
                    transforms.ColorJitter(contrast=0.5, saturation=0.5, hue=0.5),
                ]), p=0.5),

                transforms.RandomApply(torch.nn.ModuleList([
                    transforms.Grayscale(num_output_channels=3),
                ]), p=0.5),

                transforms.ToTensor(),
                #normalize
            ]
        )

    valid_transforms = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            #normalize
        ]
    )
    # Data loading code
    if args.dummy:
        print("=> Dummy CompCars data is used!")
        train_dataset = datasets.FakeData(11336, (3, 224, 224), 431, train_transforms)
        val_dataset = datasets.FakeData(4680, (3, 224, 224), 431, valid_transforms)
    else:
        traindir = os.path.join(args.data, 'image_train')
        valdir = os.path.join(args.data, 'image_valid')
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

        train_dataset = datasets.ImageFolder(
            traindir,
            train_transforms)

        val_dataset = datasets.ImageFolder(
            valdir,
            valid_transforms)

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False, drop_last=True)
    else:
        train_sampler = None
        val_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, sampler=val_sampler)

    start = time.time()
    prof = None

    if args.prof and args.gpu == 0:
        prof = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=1, repeat=1),
            record_shapes=False,
            profile_memory=False,
            with_stack=False,
        )

    for epoch in range(args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        train_acc, train_corrects, train_time, data_time, throughput = train(train_loader, model, criterion, optimizer, epoch, args)

        valid_acc, valid_corrects, valid_top1 = validate(val_loader, model, criterion, args)
        
        scheduler.step()

        print(
            "Epoch [{}/{}] train acc: {:.4f}% "
            "Valid acc: {:.4f}% Time: {:.0f}s Data time: {:.0f}s train corr: {:d}  valid corr: {:d} throughput (img/s) {:.4f} ".format(
                epoch,
                args.epochs - 1,
                train_acc,
                valid_top1,
                (train_time),
                (data_time),
                train_corrects,
                valid_corrects,
                throughput
            )
        )
        if args.name:
            log_file.write(f"{epoch},{train_acc},{valid_acc},{train_time},{train_corrects},{valid_corrects},{throughput}\n")
        if prof:
            prof.step()
    if prof:
        prof.stop()
        prof.export_chrome_trace("trace.json")
    end = time.time() - start
    print(f"Training took {end} seconds")
    if args.name:
        log_file.close()



def train(train_loader, model, criterion, optimizer, epoch, args):
    # switch to train mode
    model.train(True)

    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')

    train_running_corrects = 0
    running_loss = 0.0

    first_inputs, first_labels = None, None

    end = time.time()
    for i, (inputs, labels) in enumerate(train_loader):
        data_time.update(time.time() - end)
        #if args.gpu is not None and torch.cuda.is_available():
        #    images = images.cuda(args.gpu, non_blocking=True)
        #elif not args.gpu and torch.cuda.is_available():
        #    target = target.cuda(args.gpu, non_blocking=True)
        #elif torch.backends.mps.is_available():
        #    images = images.to('mps')
        #    target = target.to('mps')
        if args.no_memcpy:
            if i == 0:
                inputs, labels = Variable(inputs.cuda(args.gpu)), Variable(labels.cuda(args.gpu))
                first_inputs, first_labels = inputs, labels
            else:
                inputs, labels = first_inputs, first_labels
        else:
            inputs, labels = Variable(inputs.cuda(args.gpu)), Variable(labels.cuda(args.gpu))
        optimizer.zero_grad()

        # compute output
        outputs = model.forward(inputs)
        loss = criterion(outputs, labels)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(outputs, labels, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(acc1[0], inputs.size(0))
        top5.update(acc5[0], inputs.size(0))
        preds = torch.max(outputs, 1)[1]
        train_running_corrects += torch.sum(preds == labels.data)

        # compute gradient and do SGD step
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        batch_time.update(time.time() - end)
        end = time.time()

    train_epoch_acc = train_running_corrects.double() / len(train_loader.dataset)*100
    #train_time = time.time() - start

    #if args.distributed:
        #batch_time.all_reduce()

    throughput = len(train_loader.dataset) / batch_time.sum

    return train_epoch_acc, train_running_corrects, batch_time.sum, data_time.sum, throughput


def validate(val_loader, model, criterion, args):

    def run_validate(loader, base_progress=0):
        with torch.no_grad():
            end = time.time()
            for i, (images, target) in enumerate(loader):
                i = base_progress + i
                if args.gpu is not None and torch.cuda.is_available():
                    images = images.cuda(args.gpu, non_blocking=True)
                if torch.backends.mps.is_available():
                    images = images.to('mps')
                    target = target.to('mps')
                if torch.cuda.is_available():
                    target = target.cuda(args.gpu, non_blocking=True)

                # compute output
                output = model(images)
                loss = criterion(output, target)

                # measure accuracy and record loss
                acc1, acc5 = accuracy(output, target, topk=(1, 5))
                #losses.update(loss.item(), images.size(0))
                #top1.update(acc1[0], images.size(0))
                #top5.update(acc5[0], images.size(0))

                # measure elapsed time
                #batch_time.update(time.time() - end)
                end = time.time()

                #if i % args.print_freq == 0:
                #    progress.display(i + 1)

    def run_validate_2(loader):
        model.train(False)
        running_loss = 0.0
        valid_running_corrects = 0

        first_inputs, first_labels = None, None

        for i, (inputs, labels) in enumerate(loader):

            if args.no_memcpy:
                if i == 0:
                    inputs, labels = Variable(inputs.cuda(args.gpu)), Variable(labels.cuda(args.gpu))
                    first_inputs, first_labels = inputs, labels
                else:
                    inputs, labels = first_inputs, first_labels
            else:
                inputs, labels = Variable(inputs.cuda(args.gpu)), Variable(labels.cuda(args.gpu))

            #optimizer.zero_grad()

            outputs = model.forward(inputs)

            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs.data, 1)
            
            acc1, acc5 = accuracy(outputs, labels, topk=(1, 5))
            top1.update(acc1[0], inputs.size(0))
            top5.update(acc5[0], inputs.size(0))

            valid_running_corrects += torch.sum(preds == labels.data)

            running_loss += loss.item()

        valid_epoch_acc = valid_running_corrects.double() / len(loader.dataset) * 100

        return valid_epoch_acc, valid_running_corrects

    
    #losses = AverageMeter('Loss', ':.4e', Summary.NONE)
    top1 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
    top5 = AverageMeter('Acc@5', ':6.2f', Summary.AVERAGE)
    #progress = ProgressMeter(
    #    len(val_loader) + (args.distributed and (len(val_loader.sampler) * args.world_size < len(val_loader.dataset))),
    #    [batch_time, losses, top1, top5],
    #    prefix='Test: ')

    # switch to evaluate mode
    #model.eval()
    valid_acc, valid_corrects = run_validate_2(val_loader)

    if args.distributed:
        top1.all_reduce()
        top5.all_reduce()

    if args.distributed and (len(val_loader.sampler) * args.world_size < len(val_loader.dataset)):
        aux_val_dataset = Subset(val_loader.dataset,
                                 range(len(val_loader.sampler) * args.world_size, len(val_loader.dataset)))
        aux_val_loader = torch.utils.data.DataLoader(
            aux_val_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)
        valid_acc, valid_corrects = run_validate_2(aux_val_loader)

    return valid_acc, valid_corrects, top1.avg
    #return top1.avg

if __name__ == '__main__':
    main()