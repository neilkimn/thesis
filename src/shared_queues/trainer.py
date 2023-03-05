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

        torch.manual_seed(self.args.seed)

        num_ftrs = self.model.fc.in_features  # num_ftrs = 2048
        print("features", num_ftrs, "classes", 431)
        self.model.fc = torch.nn.Linear(num_ftrs, 431)

        if self.args.log_path:
            with open(self.args.log_path + f"/{self.name}.csv", "w") as f:
                f.write("epoch,train_acc,valid_acc,train_time,valid_time,train_corr,valid_corr,throughput\n")

        self.running_loss = 0.0
        self.train_running_corrects = 0
        self.model.train(True)

        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9)
        self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=7, gamma=0.1)
        self.criterion = torch.nn.CrossEntropyLoss()

        self.test_loss = 0.0
        self.test_correct = 0
        self.test_acc = 0.0

        self.train_time = 0.0
        self.validation_time = 0.0

    def send_model(self):
        self.on_device = True
        self.model.to(self.device)

    def forward(self, inputs, labels, batch_idx, epoch, pid):
        start_time = time.time()

        self.optimizer.zero_grad()
        outputs = self.model.forward(Variable(inputs))
        loss = self.criterion(outputs, Variable(labels))
        preds = torch.max(outputs, 1)[1]
        self.train_running_corrects += torch.sum(preds == labels.data)
        loss.backward()
        self.optimizer.step()
        self.running_loss += loss.item()

        self.train_time += time.time() - start_time

        if batch_idx % self.args.log_interval == 0:
            print('{}\tTrain Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                pid, epoch, batch_idx * len(inputs), self.args.train_dataset_len,
                100. * batch_idx / self.args.train_loader_len, loss.item()))
        
    def end_epoch(self, args, epoch):
        train_epoch_acc = float(self.train_running_corrects) / args.train_dataset_len * 100
        #train_time = time.time() - self.epoch_time
        train_time = self.train_time
        throughput = args.train_dataset_len / self.train_time
        train_running_corrects = self.train_running_corrects

        self.train_running_corrects = 0
        self.train_time = 0.0
        self.running_loss = 0.0
        
        pid = os.getpid()
        print(f"{pid} Training time: {train_time}s, Train accuracy: {train_epoch_acc}")
        print('{} Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            pid, self.test_loss, self.test_correct, args.valid_dataset_len,
            self.test_acc))

        if self.args.log_path:
            with open(args.log_path + f"/{self.name}.csv", "a") as f:
                f.write(f"{epoch},{train_epoch_acc},{self.test_acc},{train_time},{self.validation_time},{train_running_corrects},{self.test_correct},{throughput}\n")

        self.test_loss = 0.0
        self.test_correct = 0
        self.test_acc = 0.0
        self.validation_time = 0.0
        self.model.train(True)
        self.scheduler.step()

        #return train_epoch_acc, train_time, train_running_corrects, throughput

    def validate(self, inputs, labels):
        start_time = time.time()
        self.model.eval()

        with torch.no_grad():
            output = self.model(inputs)
            self.test_loss += self.criterion(output, labels).item()
            pred = output.max(1)[1]
            self.test_correct += pred.eq(labels).sum().item()

        self.test_loss /= self.args.valid_dataset_len
        self.test_acc = 100. * self.test_correct / self.args.valid_dataset_len
        self.validation_time += time.time() - start_time

class Trainer:
    def __init__(self, args, model, device, dataset, train_transforms, valid_transforms, shared_dataset=None):
        self.args = args

        self.device = device
        self.model = model
        num_ftrs = self.model.fc.in_features  # num_ftrs = 2048
        print("features", num_ftrs, "classes", 431)
        self.model.fc = torch.nn.Linear(num_ftrs, 431)

        self.model.to(device)

        self.train_transforms = train_transforms
        self.valid_transforms = valid_transforms
        self.dataset = dataset

        self.shared_dataset = shared_dataset
        #self.model.share_memory()

        if self.args.log_path:
            self.args.log_path = Path(self.args.log_path)
            self.args.log_path.mkdir(parents=True, exist_ok=True)
            with open(self.args.log_path / f"{self.model.name}.csv", "w") as f:
                f.write("epoch,train_acc,valid_acc,train_time,valid_time,train_corr,valid_corr,throughput\n")
            self.gpu_path = self.args.log_path / f"{self.model.name}_gpu_util.csv"
            os.system(f"nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,nounits -f {self.gpu_path}")

        """Split train and test"""
        train_len = int(0.7 * len(dataset))
        valid_len = len(dataset) - train_len
        train_set, valid_set = D.random_split(dataset, lengths=[train_len, valid_len], generator=torch.Generator().manual_seed(42))

        self.train_set = DatasetFromSubset(train_set, self.train_transforms)
        self.valid_set = DatasetFromSubset(valid_set, self.valid_transforms)

        torch.manual_seed(self.args.seed)
        self.loader_valid = D.DataLoader(
            self.valid_set,
            batch_size=80,
            shuffle=True,
            num_workers=self.args.num_workers,
            pin_memory=True,
            prefetch_factor=2,
        )

        self.criterion = torch.nn.CrossEntropyLoss()

    def train(self, rank, lock=None):
        torch.manual_seed(self.args.seed)

        optimizer = optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
        criterion = torch.nn.CrossEntropyLoss()

        train_loader = D.DataLoader(
            self.train_set,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=self.args.num_workers,
            pin_memory=True,
            prefetch_factor=2,
        )

        for epoch in range(1, self.args.epochs + 1):
            train_acc, train_time, train_corrects, throughput = self.train_epoch(epoch, train_loader, optimizer, scheduler, criterion, rank, lock)

            valid_acc, valid_corrects = 0, 0
            #if epoch == self.args.epochs:
            valid_acc, valid_corrects, valid_time = self.test(rank, epoch)

            if self.args.log_path:
                with open(self.args.log_path / f"{self.model.name}.csv", "a") as f:
                    f.write(f"{epoch},{train_acc},{valid_acc},{train_time},{valid_time},{train_corrects},{valid_corrects},{throughput}\n")
                os.system(f"nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader >> {self.gpu_path}")



    def train_epoch(self, epoch, train_loader, optimizer, scheduler, criterion, rank, lock=None):
        epoch_time = time.time()

        scheduler.step()
        self.model.train(True)
        pid = os.getpid()
        running_loss = 0.0
        train_running_corrects = 0

        if self.args.debug_data_dir:
            debug_indices = Path(self.args.debug_data_dir) / f"rank_{str(max(rank,0))}_epoch_{epoch}" / "indices.txt"
            debug_indices.parent.mkdir(parents=True, exist_ok=True)

        loader_iter = iter(train_loader)
        for batch_idx in range(len(train_loader)):
            if self.shared_dataset:
                accs = self.shared_dataset.get_accesses(lock, batch_idx)
                if accs:
                    (inputs, labels) = self.shared_dataset.get_batch(lock, batch_idx, pid)
                else:
                    (inputs, labels, _) = next(loader_iter)
                    inputs, labels = Variable(inputs.to(self.device)), Variable(labels.to(self.device))
                    self.shared_dataset.set_batch(lock, inputs, labels, batch_idx, pid)
            else:
                (inputs, labels, indices) = next(loader_iter)
                #print(f"indices: {indices[:2]} ... {indices[-2:]}")
                inputs, labels = Variable(inputs.to(self.device)), Variable(labels.to(self.device))

            if self.args.debug_data_dir:
                with open(debug_indices, "a") as f:
                    f.write(" ".join(list(map(str, indices.tolist()))))
                    f.write("\n")

            optimizer.zero_grad()

            outputs = self.model.forward(inputs)

            loss = criterion(outputs, labels)
            preds = torch.max(outputs, 1)[1]

            train_running_corrects += torch.sum(preds == labels.data)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if batch_idx % self.args.log_interval == 0:
                print('{}\tTrain Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    pid, epoch, batch_idx * len(inputs), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))

            if self.shared_dataset:
                self.shared_dataset.remove_batch(lock, batch_idx, pid)

        train_epoch_acc = train_running_corrects.double() / len(train_loader.dataset) * 100

        train_time = time.time() - epoch_time

        throughput = len(train_loader.dataset) / train_time

        print(f"{pid} Training time: {train_time}s, Train accuracy: {train_epoch_acc}")

        return train_epoch_acc, train_time, train_running_corrects, throughput

    def test(self, rank, epoch):
        pid = os.getpid()
        self.model.eval()
        test_loss = 0
        correct = 0
        epoch_time = time.time()

        if self.args.debug_data_dir:
            debug_indices = Path(self.args.debug_data_dir) / f"rank_{str(max(rank,0))}_epoch_{epoch}" / "val_indices.txt"
            debug_indices.parent.mkdir(parents=True, exist_ok=True)

        with torch.no_grad():
            for data, target, indices in self.loader_valid:
                #print(f"indices: {indices[:2]} ... {indices[-2:]}")
                output = self.model(data.to(self.device))
                test_loss += self.criterion(output, target.to(self.device)).item()
                pred = output.max(1)[1]
                correct += pred.eq(target.to(self.device)).sum().item()

                if self.args.debug_data_dir:
                    with open(debug_indices, "a") as f:
                        f.write(" ".join(list(map(str, indices.tolist()))))
                        f.write("\n")

        test_loss /= len(self.loader_valid.dataset)
        test_acc = 100. * correct / len(self.loader_valid.dataset)
        print('{} Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            pid, test_loss, correct, len(self.loader_valid.dataset),
            test_acc))
        
        test_time = time.time() - epoch_time
        
        return test_acc, correct, test_time