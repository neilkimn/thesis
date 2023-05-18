import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils import data as D
from torch.autograd import Variable

import os
import time
from pathlib import Path

from shared_queues.dataset import DatasetFromSubset

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

        if self.args.dataset == "compcars":
            num_ftrs = self.model.fc.in_features  # num_ftrs = 2048
            self.model.fc = torch.nn.Linear(num_ftrs, 431)

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
                shuffle=True,
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