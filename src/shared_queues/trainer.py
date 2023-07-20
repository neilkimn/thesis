from typing import Any
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils import data as D
from torch.autograd import Variable
import torch.nn as nn

import os
import sys
import math
import time
from pathlib import Path
import logging
from shared_queues.dataset import DatasetFromSubset
#from detectron2.solver import build_lr_scheduler, build_optimizer
#from detectron2.utils.events import EventStorage
from timm import utils
from timm.optim import create_optimizer_v2, optimizer_kwargs
from timm.scheduler import create_scheduler_v2, scheduler_kwargs
from contextlib import suppress
from torch_utils import utils as coco_utils

class TimmTrainer:
    def __init__(self, args, model, device, name):
        self.args = args
        self.model = model
        self.device = device
        self.name = name
        self.on_device = False
        self.log_name = self.name + f"_bs{args.batch_size}_{args.training_workers}tw_{args.validation_workers}vw_{args.prefetch_factor}pf"
        self.model.train(True)
        self.current_epoch = 0
        self.update_idx = 0
        self.train_running_corrects = 0
        self.valid_running_corrects = 0

        if self.args.seed:
            torch.manual_seed(self.args.seed)

        self.optimizer = create_optimizer_v2(
            self.model,
            **optimizer_kwargs(cfg=self.args),
            **args.opt_kwargs
        )

        self.accum_steps = 1
        
        self.train_loss_fn = nn.CrossEntropyLoss()
        self.train_loss_fn = self.train_loss_fn.to(device=device)
        self.validate_loss_fn = nn.CrossEntropyLoss().to(device=device)
        self.logger = logging.getLogger('train')

    def init_log(self, pid):
        self.log_name += f"_pid_{pid}"
        log_path = self.args.log_dir + "/" + self.log_name + ".csv"
        with open(log_path, "w") as f:
            f.write("timestamp,epoch,train_acc,valid_acc,train_time,batch_time,valid_time,total_time,train_corr,valid_corr,throughput\n")
            f.write(f"{int(time.time())},0,0.0,0.0,0.0,0.0,0.0,0.0,0,0,0\n")
        gpu_path = self.args.log_dir + "/" + self.log_name + "_gpu_util.csv"
        os.system(f"nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,nounits -f {gpu_path}")
        return log_path, gpu_path

    def set_lr_scheduler(self, updates_per_epoch):
        self.lr_scheduler, self.num_epochs = create_scheduler_v2(
            self.optimizer,
            **scheduler_kwargs(self.args),
            updates_per_epoch=updates_per_epoch,
        )

    def send_model(self, channels_last):
        self.on_device = True
        self.model.to(self.device)
        if channels_last:
            self.model.to(self.device, memory_format=torch.channels_last)

    def start_epoch(self, epoch, loader_len):
        self.second_order = hasattr(self.optimizer, 'is_second_order') and self.optimizer.is_second_order
        self.has_no_sync = hasattr(self.model, "no_sync")
        self.update_time_m = utils.AverageMeter()
        self.data_time_m = utils.AverageMeter()
        self.losses_m = utils.AverageMeter()

        self.batch_time_m = utils.AverageMeter()
        self.val_losses_m = utils.AverageMeter()
        self.top1_m = utils.AverageMeter()
        self.top5_m = utils.AverageMeter()

        self.last_accum_steps = loader_len % self.accum_steps
        self.updates_per_epoch = (loader_len + self.accum_steps - 1) // self.accum_steps
        self.num_updates = epoch * self.updates_per_epoch
        self.last_batch_idx = loader_len - 1
        self.last_batch_idx_to_accum = loader_len - self.last_accum_steps

        self.data_start_time = self.update_start_time = time.time()
        self.optimizer.zero_grad()
        self.update_sample_count = 0
        self.current_epoch = epoch

    def _forward(self, inputs, labels):
        with suppress():
            outputs = self.model(inputs)
            loss = self.train_loss_fn(outputs, labels)

        preds = torch.max(outputs, 1)[1]
        self.train_running_corrects += torch.sum(preds == labels.data)

        return loss
    
    def _backward(self, loss, need_update):
        loss.backward()
        if need_update:
            self.optimizer.step()

    def forward(self, index, inputs, labels):
        last_batch = index == self.last_batch_idx
        need_update = last_batch or (index + 1) % self.accum_steps == 0
        self.update_idx = index // self.accum_steps
        if index >= self.last_batch_idx_to_accum:
            self.accum_steps = self.last_accum_steps
        if self.args.channels_last:
            inputs = inputs.contiguous(memory_format=torch.channels_last)
        self.data_time_m.update(self.accum_steps * (time.time() - self.data_start_time))

        if self.has_no_sync and not need_update:
            with self.model.no_sync():
                loss = self._forward(inputs, labels)
                self._backward(loss, need_update)
        else:
            loss = self._forward(inputs, labels)
            self._backward(loss, need_update)

        self.losses_m.update(loss.item() * self.accum_steps, inputs.size(0))
        self.update_sample_count += inputs.size(0)

        if not need_update:
            data_start_time = time.time()
            return

        self.num_updates += 1
        self.optimizer.zero_grad()

        time_now = time.time()
        self.update_time_m.update(time.time() - self.update_start_time)
        self.update_start_time = time_now

        return loss
    
    def validate(self, inputs, labels):
        self.model.eval()
        with torch.no_grad():
            if self.args.channels_last:
                inputs = inputs.contiguous(memory_format=torch.channels_last)
            output = self.model(inputs)
            preds = torch.max(output, 1)[1]
            self.valid_running_corrects += torch.sum(preds == labels.data)
            if isinstance(output, (tuple, list)):
                output = output[0]
            loss = self.validate_loss_fn(output, labels)
            acc1, acc5 = utils.accuracy(output, labels, topk=(1, 5))
            reduced_loss = loss.data

            if self.args.device.type == 'cuda':
                torch.cuda.synchronize()
            self.losses_m.update(reduced_loss.item(), inputs.size(0))
            self.top1_m.update(acc1.item(), output.size(0))
            self.top5_m.update(acc5.item(), output.size(0))
        return self.losses_m.avg, self.top1_m.avg, self.valid_running_corrects

    def end_epoch(self, args):
        train_epoch_acc = float(self.train_running_corrects) / args.train_dataset_len * 100
        train_running_corrects = self.train_running_corrects
        self.train_running_corrects = 0
        self.valid_running_corrects = 0

        print(
            f'Train: {self.current_epoch} [{self.update_idx}/{self.updates_per_epoch} '
            f'({100. * self.update_idx / (self.updates_per_epoch - 1)}%)]  '
            f'Loss: {self.losses_m.val} ({self.losses_m.avg})  '
            f'Time: {self.update_time_m.val}s,'
            #f'({self.update_time_m.avg:.3f}s, {self.update_sample_count / self.update_time_m.avg:>7.2f}/s)  '
            #f'LR: {lr:.3e}  '
            f'Data: {self.data_time_m.val} ({self.data_time_m.avg})'
        )

        print(
            f'Validation:'
            f'Loss: {self.losses_m.val} ({self.losses_m.avg})  '
            f'Acc@1: {self.top1_m.val} ({self.top1_m.avg})  '
            f'Acc@5: {self.top5_m.val} ({self.top5_m.avg})'
        )
        return train_epoch_acc, train_running_corrects

import torchvision
from torch_utils import utils as coco_trainer_utils

class CocoTrainer:
    def __init__(self, args, model, device, name):
        self.args = args
        self.device = device
        self.model = model
        self.name = name
        self.on_device = False
        self.log_name = self.name + f"_bs{args.batch_size}_{args.training_workers}tw_{args.validation_workers}vw_{args.prefetch_factor}pf"

        self.current_epoch = 0
        self.train_running_corrects = 0
        self.valid_running_corrects = 0

        param_groups = torchvision.ops._utils.split_normalization_params(model)
        wd_groups = [args.norm_weight_decay, args.weight_decay]
        parameters = [{"params": p, "weight_decay": w} for p, w in zip(param_groups, wd_groups) if p]
        opt_name = args.opt.lower()
        if opt_name.startswith("sgd"):
            self.optimizer = torch.optim.SGD(
                parameters,
                lr=args.lr,
                momentum=args.momentum,
                weight_decay=args.weight_decay,
                nesterov="nesterov" in opt_name,
            )
        elif opt_name == "adamw":
            self.optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=args.weight_decay)
        else:
            raise RuntimeError(f"Invalid optimizer {args.opt}. Only SGD and AdamW are supported.")

    def send_model(self):
        self.on_device = True
        self.model.to(self.device)

    def set_epoch_lr_scheduler(self, lr_scheduler):
        self.epoch_lr_scheduler = lr_scheduler

    def init_log(self, pid):
        self.log_name += f"_pid_{pid}"
        log_path = self.args.log_dir + "/" + self.log_name + ".csv"
        with open(log_path, "w") as f:
            f.write("timestamp,epoch,train_acc,valid_acc,train_time,batch_time,valid_time,total_time,train_corr,valid_corr,throughput\n")
            f.write(f"{int(time.time())},0,0.0,0.0,0.0,0.0,0.0,0.0,0,0,0\n")
        gpu_path = self.args.log_dir + "/" + self.log_name + "_gpu_util.csv"
        os.system(f"nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,nounits -f {gpu_path}")
        return log_path, gpu_path
    
    def start_epoch(self, epoch, data_loader_len):
        print("Start epoch called")
        self.model.train()
        self.lr_scheduler = None
        if epoch == 0:
            warmup_factor = 1.0 / 1000
            warmup_iters = min(1000, data_loader_len - 1)

            self.lr_scheduler = torch.optim.lr_scheduler.LinearLR(
                self.optimizer, start_factor=warmup_factor, total_iters=warmup_iters
            )
        

    def forward(self, inputs, targets):
        loss_dict = self.model(inputs, targets)
        losses = sum(loss for loss in loss_dict.values())
        loss_dict_reduced = coco_utils.reduce_dict(loss_dict)

        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()
        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}")
            #print(loss_dict_reduced)
            #sys.exit(1)
        self.optimizer.zero_grad()

        losses.backward()
        self.optimizer.step()

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        return loss_value


    def end_epoch(self):
        self.epoch_lr_scheduler.step()
        return 0,0


class ProcTrainer:
    def __init__(self, args, model, device):
        self.args = args
        self.device = device
        self.model = model
        self.name = model.name
        self.on_device = False
        self.log_name = self.name + f"_bs{args.batch_size}_{args.training_workers}tw_{args.validation_workers}vw_{args.prefetch_factor}pf"

        if self.args.seed:
            torch.manual_seed(self.args.seed)

        self.running_loss = 0.0
        self.train_running_corrects = 0
        self.model.train(True)

        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9)
        self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=7, gamma=0.1)
        self.criterion = torch.nn.CrossEntropyLoss()

        self.val_loss = 0.0
        self.val_correct = 0
        self.val_acc = 0.0

    def init_log(self, pid):
        self.log_name += f"_pid_{pid}"
        log_path = self.args.log_dir + "/" + self.log_name + ".csv"
        with open(log_path, "w") as f:
            f.write("timestamp,epoch,train_acc,valid_acc,train_time,batch_time,valid_time,total_time,train_corr,valid_corr,throughput\n")
            f.write(f"{int(time.time())},0,0.0,0.0,0.0,0.0,0.0,0.0,0,0,0\n")
        gpu_path = self.args.log_dir + "/" + self.log_name + "_gpu_util.csv"
        os.system(f"nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,nounits -f {gpu_path}")
        return log_path, gpu_path

    def send_model(self):
        self.on_device = True
        self.model.to(self.device)

    def forward(self, inputs, labels):

        self.optimizer.zero_grad()
        outputs = self.model.forward(Variable(inputs))
        loss = self.criterion(outputs, Variable(labels))
        preds = torch.max(outputs, 1)[1]
        self.train_running_corrects += torch.sum(preds == labels.data)
        loss.backward()
        self.optimizer.step()
        self.running_loss += loss.item()

        return loss
    
    def end_epoch(self, args):
        train_epoch_acc = float(self.train_running_corrects) / args.train_dataset_len * 100
        train_running_corrects = self.train_running_corrects
        self.train_running_corrects = 0

        self.val_loss = 0.0
        self.val_correct = 0
        self.val_acc = 0.0

        self.model.train(True)
        self.scheduler.step()

        return train_epoch_acc, train_running_corrects

    def validate(self, inputs, labels):
        self.model.eval()

        with torch.no_grad():
            output = self.model(inputs)
            self.val_loss += self.criterion(output, labels).item()
            pred = output.max(1)[1]
            self.val_correct += pred.eq(labels).sum().item()

        self.val_loss /= self.args.valid_dataset_len
        self.val_acc = 100. * self.val_correct / self.args.valid_dataset_len

        return self.val_loss, self.val_acc, self.val_correct
        
    
class RCNNProcTrainer:
    def __init__(self, args, model, device, name):
        self.args = args
        self.device = device
        self.model = model
        self.name = name
        self.on_device = False
        self.log_name = self.name + f"_bs{args.batch_size}_{args.training_workers}tw_{args.validation_workers}vw_{args.prefetch_factor}pf"

        if self.args.seed:
            torch.manual_seed(self.args.seed)

        self.running_loss = 0.0
        self.train_running_corrects = 0
        self.model.train(True)

        params = [p for p in model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.SGD(params, lr=0.001, momentum=0.9, nesterov=True)

        warmup_factor = 1.0 / 1000
        warmup_iters = 1000
        self.scheduler = torch.optim.lr_scheduler.LinearLR(
            self.optimizer, start_factor=warmup_factor, total_iters=warmup_iters
        )

        self.val_loss = 0.0
        self.val_correct = 0
        self.val_acc = 0.0
        self.train_loss_list = []
        self.loss_cls_list = []
        self.loss_box_reg_list = []
        self.loss_objectness_list = []
        self.loss_rpn_list = []
        self.train_loss_list_epoch = []
        self.val_map_05 = []
        self.val_map = []
        self.start_epochs = 0

    def init_log(self, pid):
        self.log_name += f"_pid_{pid}"
        log_path = self.args.log_dir + "/" + self.log_name + ".csv"
        with open(log_path, "w") as f:
            f.write("timestamp,epoch,train_acc,valid_acc,train_time,batch_time,valid_time,total_time,train_corr,valid_corr,throughput\n")
            f.write(f"{int(time.time())},0,0.0,0.0,0.0,0.0,0.0,0.0,0,0,0\n")
        gpu_path = self.args.log_dir + "/" + self.log_name + "_gpu_util.csv"
        os.system(f"nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,nounits -f {gpu_path}")
        return log_path, gpu_path

    def send_model(self):
        self.on_device = True
        self.model.to(self.device)

    def forward(self, images, targets):
        loss_dict = self.model(images, targets)
        return loss_dict
        
    def end_epoch(self, args):
        train_epoch_acc = float(self.train_running_corrects) / args.train_dataset_len * 100
        train_running_corrects = self.train_running_corrects
        self.train_running_corrects = 0

        self.val_loss = 0.0
        self.val_correct = 0
        self.val_acc = 0.0

        self.model.train(True)
        self.scheduler.step()

        return train_epoch_acc, train_running_corrects

    def validate(self, inputs, labels):
        self.model.eval()

        with torch.no_grad():
            output = self.model(inputs)
            self.val_loss += self.criterion(output, labels).item()
            pred = output.max(1)[1]
            self.val_correct += pred.eq(labels).sum().item()

        self.val_loss /= self.args.valid_dataset_len
        self.val_acc = 100. * self.val_correct / self.args.valid_dataset_len

        return self.val_loss, self.val_acc, self.val_correct
    
    
class NaiveTrainer:
    def __init__(self, args, model, device, train_dataset, val_dataset):
        self.args = args
        self.device = device
        self.model = model
        self.name = model.name
        self.pid = os.getpid()

        if self.args.seed:
            torch.manual_seed(self.args.seed)

        if args.dataset == "compcars":
            num_ftrs = self.model.fc.in_features  # num_ftrs = 2048
            self.model.fc = torch.nn.Linear(num_ftrs, 431)

        self.running_loss = 0.0
        self.train_running_corrects = 0

        if self.args.log_path:
            self.args.log_path = Path(self.args.log_path)
            self.args.log_path.mkdir(parents=True, exist_ok=True)
            with open(self.args.log_path / f"{self.model.name}_pid_{self.pid}.csv", "w") as f:
                f.write("timestamp,epoch,train_acc,valid_acc,train_time,batch_time,valid_time,total_time,train_corr,valid_corr,throughput\n")
                f.write(f"{int(time.time())},0,0.0,0.0,0.0,0.0,0.0,0.0,0,0,0\n")
            self.gpu_path = self.args.log_path / f"{self.model.name}_pid_{self.pid}_gpu_util.csv"
            os.system(f"nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,nounits -f {self.gpu_path}")

        self.model.to(self.device)

        """Split train and test"""
        self.train_set = train_dataset
        self.valid_set = val_dataset

        self.loader_train = D.DataLoader(
            self.train_set,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=self.args.training_workers,
            pin_memory=True,
            prefetch_factor=self.args.prefetch_factor,
        )

        self.loader_valid = D.DataLoader(
            self.valid_set,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=self.args.validation_workers,
            pin_memory=True,
            prefetch_factor=self.args.prefetch_factor,
        )

        self.train_dataset_len = len(self.loader_train.dataset)
        self.train_loader_len = len(self.loader_train)

        self.valid_dataset_len = len(self.loader_valid.dataset)
        self.valid_loader_len = len(self.loader_valid)
        
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9)
        self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=7, gamma=0.1)
        self.criterion = torch.nn.CrossEntropyLoss()

        self.val_loss = 0.0
        self.val_correct = 0
        self.val_acc = 0.0

    def train(self, rank):
        if self.args.seed:
            torch.manual_seed(self.args.seed)

        epoch_start = time.time()
        for epoch in range(1, self.args.epochs + 1):
            train_acc, train_time, train_batch_time, train_corrects, throughput = self.train_epoch(epoch)

            valid_time, val_batch_time, valid_corrects = 0, 0, 0

            if epoch > 10:
                #valid_acc, valid_corrects, valid_time, val_batch_time = self.test(rank, epoch)
                self.val_acc, self.val_correct, valid_time, val_batch_time = self.test(rank, epoch)
            total_batch_time = train_batch_time + val_batch_time

            total_time = time.time() - epoch_start
            epoch_start = time.time()

            if self.args.log_path:
                with open(self.args.log_path / f"{self.model.name}_pid_{self.pid}.csv", "a") as f:
                    f.write(f"{int(time.time())},{epoch},{train_acc},{self.val_acc},{train_time},{valid_time},{total_batch_time},{total_time},{train_corrects},{valid_corrects},{throughput}\n")
                os.system(f"nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader >> {self.gpu_path}")

    def forward(self, inputs, labels):

        self.optimizer.zero_grad()
        outputs = self.model.forward(Variable(inputs))
        loss = self.criterion(outputs, Variable(labels))
        preds = torch.max(outputs, 1)[1]
        self.train_running_corrects += torch.sum(preds == labels.data)
        loss.backward()
        self.optimizer.step()
        self.running_loss += loss.item()

        return loss
    
    def test(self, epoch):
        self.model.eval()
        epoch_time = time.time()
        
        start_time = time.time()
        for batch_idx, inputs, labels in enumerate(self.loader_valid):
            pass
    
    def validate(self, inputs, labels):
        self.model.eval()

        with torch.no_grad():
            output = self.model(inputs)
            self.val_loss += self.criterion(output, labels).item()
            pred = output.max(1)[1]
            self.val_correct += pred.eq(labels).sum().item()

        self.val_loss /= self.valid_dataset_len
        self.val_acc = 100. * self.val_correct / self.valid_dataset_len

        return self.val_loss, self.val_acc, self.val_correct
    
    def end_epoch(self):
        train_epoch_acc = float(self.train_running_corrects) / self.train_dataset_len * 100
        train_running_corrects = self.train_running_corrects
        self.train_running_corrects = 0

        self.val_loss = 0.0
        self.val_correct = 0
        self.val_acc = 0.0

        self.model.train(True)
        self.scheduler.step()

        return train_epoch_acc, train_running_corrects
    
    def train_epoch(self, epoch):
        pid = os.getpid()
        epoch_time = time.time()

        self.model.train(True)
        running_loss = 0.0
        running_time = epoch_time
        train_running_corrects = 0
        batch_time = 0

        start_time = time.time()
        for batch_idx, (inputs, labels) in enumerate(self.loader_train):
            end_time = time.time() - start_time
            batch_time += end_time
            inputs, labels = Variable(inputs.to(self.device)), Variable(labels.to(self.device))
            loss = self.forward(inputs, labels)

            running_loss += loss.item()

            if batch_idx % self.args.log_interval == 0:
                print('{}\tTrain Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} Throughput [img/s]: {:.1f}'.format(
                    pid, epoch, batch_idx * len(inputs), len(self.loader_train.dataset),
                    100. * batch_idx / len(self.loader_train), loss.item(), (batch_idx * len(inputs) / (time.time() - epoch_time)) ))
                if self.args.log_path:
                    with open(self.args.log_path / f"{self.model.name}_pid_{self.pid}_output.csv", "a") as f:
                        f.write(f"{pid}\tTrain Epoch: {epoch} [{batch_idx * len(inputs)}/{len(self.loader_train.dataset)} ({round(100. * batch_idx / len(self.loader_train),2)}%)]\tLoss: {round(loss.item(),2)} Throughput [img/s]: {round(batch_idx * len(inputs) / (time.time() - epoch_time), 2)}\n")
            
            start_time = time.time()
        
        train_epoch_acc, train_running_corrects = self.end_epoch()

        epoch_time = time.time() - epoch_time
        train_time = epoch_time - batch_time

        throughput = len(self.loader_train.dataset) / epoch_time

        print(f"{pid} Training time: {epoch_time}s, Train accuracy: {train_epoch_acc}")

        return train_epoch_acc, train_time, batch_time, train_running_corrects, throughput


class Trainer:
    def __init__(self, args, model, device, train_dataset=None, val_dataset=None, train_loader=None, valid_loader=None):
        self.args = args

        self.device = device
        self.model = model
        self.dali = False
        if args.dataset == "compcars":
            num_ftrs = self.model.fc.in_features  # num_ftrs = 2048
            self.model.fc = torch.nn.Linear(num_ftrs, 431)

        self.model.to(device)

        self.pid = os.getpid()

        if self.args.log_path:
            self.args.log_path = Path(self.args.log_path)
            self.args.log_path.mkdir(parents=True, exist_ok=True)
            self.args.log_name = f"{self.model.name}_bs{self.args.batch_size}_{self.args.training_workers}tw_{self.args.validation_workers}vw_pid_{self.pid}"
            with open(self.args.log_path / f"{self.args.log_name}.csv", "w") as f:
                f.write("timestamp,epoch,train_acc,valid_acc,train_time,batch_time,valid_time,total_time,train_corr,valid_corr,throughput\n")
                f.write(f"{int(time.time())},0,0.0,0.0,0.0,0.0,0.0,0.0,0,0,0\n")
            self.gpu_path = self.args.log_path / f"{self.args.log_name}_gpu_util.csv"
            os.system(f"nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,nounits -f {self.gpu_path}")

        """Split train and test"""
        self.train_set = train_dataset
        self.valid_set = val_dataset

        if self.args.seed:
            torch.manual_seed(self.args.seed)
        if train_dataset and val_dataset:
            self.train_loader = D.DataLoader(
                self.train_set,
                batch_size=self.args.batch_size,
                shuffle=True,
                num_workers=self.args.training_workers,
                pin_memory=True,
                prefetch_factor=self.args.prefetch_factor,
                persistent_workers=True,
            )
            self.valid_loader = D.DataLoader(
                self.valid_set,
                batch_size=self.args.batch_size,
                shuffle=False,
                num_workers=self.args.validation_workers,
                pin_memory=True,
                prefetch_factor=self.args.prefetch_factor,
            )
        if train_loader:
            self.train_loader = train_loader
            self.valid_loader = valid_loader
            self.dali = True
        
        self.criterion = torch.nn.CrossEntropyLoss()


    def train(self, rank):
        if self.args.seed:
            torch.manual_seed(self.args.seed)

        optimizer = optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
        criterion = torch.nn.CrossEntropyLoss()
        
        epoch_start = time.time()
        for epoch in range(1, self.args.epochs + 1):
            train_acc, train_time, train_batch_time, train_corrects, throughput = self.train_epoch(epoch, self.train_loader, optimizer, scheduler, criterion)

            valid_acc, valid_corrects, valid_time, val_batch_time = 0, 0, 0, 0
            #if epoch == self.args.epochs:
            if epoch > 10:
                valid_acc, valid_corrects, valid_time, val_batch_time = self.test(rank, epoch)
            total_batch_time = train_batch_time + val_batch_time
            #total_time = train_time + valid_time + total_batch_time
            total_time = time.time() - epoch_start
            epoch_start = time.time()

            if self.args.log_path:
                with open(self.args.log_path / f"{self.args.log_name}.csv", "a") as f:
                    f.write(f"{int(time.time())},{epoch},{train_acc},{valid_acc},{train_time},{total_batch_time},{valid_time},{total_time},{train_corrects},{valid_corrects},{throughput}\n")
                os.system(f"nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader >> {self.gpu_path}")

    def train_epoch(self, epoch, train_loader, optimizer, scheduler, criterion):
        pid = os.getpid()
        epoch_time = time.time()

        scheduler.step()
        self.model.train(True)
        pid = os.getpid()
        running_loss = 0.0
        running_time = epoch_time
        train_running_corrects = 0
        batch_time = 0

        if self.args.dummy_data:
            loader_iter = iter(train_loader)
            inputs, labels = next(loader_iter)
            inputs, labels = Variable(inputs.to(self.device)), Variable(labels.to(self.device))

        if self.args.debug_data_dir:
            debug_indices = Path(self.args.debug_data_dir) / f"{pid}_epoch_{epoch}" / "indices.txt"
            debug_indices.parent.mkdir(parents=True, exist_ok=True)

        if not self.dali:
            loader_iter = iter(train_loader)

            for batch_idx in range(len(train_loader)):
                start_time = time.time()
                if not self.args.dummy_data:
                    (inputs, labels) = next(loader_iter)
                    inputs, labels = Variable(inputs.to(self.device)), Variable(labels.to(self.device))
                end_time = time.time() - start_time
                batch_time += end_time

                optimizer.zero_grad()

                outputs = self.model.forward(inputs)

                loss = criterion(outputs, labels)
                preds = torch.max(outputs, 1)[1]

                train_running_corrects += torch.sum(preds == labels.data)

                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                if batch_idx % self.args.log_interval == 0:
                    print('{}\tTrain Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} Throughput [img/s]: {:.1f} Data time {:.1f}'.format(
                        pid, epoch, batch_idx * len(inputs), len(train_loader.dataset),
                        100. * batch_idx / len(train_loader), loss.item(), (batch_idx * len(inputs) / (time.time() - epoch_time)), batch_time ))
                    if self.args.log_path:
                        with open(self.args.log_path / f"{self.args.log_name}_output.csv", "a") as f:
                            f.write(f"{pid}\tTrain Epoch: {epoch} [{batch_idx * len(inputs)}/{len(train_loader.dataset)} ({round(100. * batch_idx / len(train_loader),2)}%)]\tLoss: {round(loss.item(),2)} Throughput [img/s]: {round(batch_idx * len(inputs) / (time.time() - epoch_time), 2)}\n")
        
        else:
            start_time = time.time()
            for batch_idx, data in enumerate(self.train_loader.dataset):
                inputs, labels = data[0]["data"], data[0]["label"]
                labels = labels.long()
                inputs, labels = Variable(inputs.to(self.device)), Variable(labels.to(self.device))

                end_time = time.time() - start_time
                batch_time += end_time

                optimizer.zero_grad()

                outputs = self.model.forward(inputs)

                loss = criterion(outputs, labels)
                preds = torch.max(outputs, 1)[1]

                train_running_corrects += torch.sum(preds == labels.data)

                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                if batch_idx % self.args.log_interval == 0:
                    print('{}\tTrain Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} Throughput [img/s]: {:.1f} Data time {:.1f}'.format(
                        pid, epoch, batch_idx * len(inputs), len(train_loader),
                        100. * batch_idx / len(train_loader), loss.item(), (batch_idx * len(inputs) / (time.time() - epoch_time)), batch_time ))
                    if self.args.log_path:
                        with open(self.args.log_path / f"{self.args.log_name}_output.csv", "a") as f:
                            f.write(f"{pid}\tTrain Epoch: {epoch} [{batch_idx * len(inputs)}/{len(train_loader.dataset)} ({round(100. * batch_idx / len(train_loader),2)}%)]\tLoss: {round(loss.item(),2)} Throughput [img/s]: {round(batch_idx * len(inputs) / (time.time() - epoch_time), 2)}\n")
                start_time = time.time()


        train_epoch_acc = train_running_corrects.double() / len(train_loader) * 100

        epoch_time = time.time() - epoch_time
        train_time = epoch_time - batch_time

        throughput = len(train_loader.dataset) / epoch_time

        print(f"{pid} Training time: {epoch_time}s, Train accuracy: {train_epoch_acc}")

        return train_epoch_acc, train_time, batch_time, train_running_corrects, throughput

    def test(self, rank, epoch):
        pid = os.getpid()
        self.model.eval()
        test_loss = 0
        correct = 0
        epoch_time = time.time()
        batch_time = 0

        if self.args.debug_data_dir:
            debug_indices = Path(self.args.debug_data_dir) / f"{pid}_rank_{str(max(rank,0))}_epoch_{epoch}" / "val_indices.txt"
            debug_indices.parent.mkdir(parents=True, exist_ok=True)

        with torch.no_grad():
            if not self.dali:
                loader_iter = iter(self.valid_loader)

                for batch_idx in range(len(self.valid_loader)):
                    start_time = time.time()
                    
                    (inputs, target) = next(loader_iter)
                    
                    end_time = time.time() - start_time
                    batch_time += end_time

                    output = self.model(inputs.to(self.device))
                    test_loss += self.criterion(output, target.to(self.device)).item()
                    pred = output.max(1)[1]
                    correct += pred.eq(target.to(self.device)).sum().item()
            else:
                for data in self.valid_loader.dataset:
                    start_time = time.time()
                    
                    inputs, labels = data[0]["data"], data[0]["label"]
                    labels = labels.long()
                    
                    end_time = time.time() - start_time
                    batch_time += end_time

                    output = self.model(inputs.to(self.device))
                    test_loss += self.criterion(output, labels.to(self.device)).item()
                    pred = output.max(1)[1]
                    correct += pred.eq(labels.to(self.device)).sum().item()

        if not self.dali:
            test_loss /= len(self.valid_loader.dataset)
            test_acc = 100. * correct / len(self.valid_loader.dataset)
            print('{} Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
                pid, test_loss, correct, len(self.valid_loader.dataset),
                test_acc))

            epoch_time = time.time() - epoch_time
            test_time = epoch_time - batch_time
        else:
            test_loss /= len(self.valid_loader)
            test_acc = 100. * correct / len(self.valid_loader)
            print('{} Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
                pid, test_loss, correct, len(self.valid_loader),
                test_acc))

            epoch_time = time.time() - epoch_time
            test_time = epoch_time - batch_time
        
        return test_acc, correct, test_time, batch_time