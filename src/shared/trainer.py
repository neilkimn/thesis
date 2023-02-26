import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils import data as D
from torch.autograd import Variable

from shared.dataset import DatasetFromSubset

import os
import time

class Trainer:
    def __init__(self, args, manager, model, device, dataset, train_transforms, valid_transforms, shared_dataset):
        self.args = args
        self.manager = manager

        self.device = device
        self.model = model
        num_ftrs = self.model.fc.in_features  # num_ftrs = 2048
        print("features", num_ftrs, "classes", 431)
        self.model.fc = torch.nn.Linear(num_ftrs, 431)

        self.model.to(device)
        self.model.share_memory()

        self.train_transforms = train_transforms
        self.valid_transforms = valid_transforms
        self.dataset = dataset

        self.shared_dataset = shared_dataset

        """Split train and test"""
        train_len = int(0.7 * len(dataset))
        valid_len = len(dataset) - train_len
        train_set, valid_set = D.random_split(dataset, lengths=[train_len, valid_len], generator=torch.Generator().manual_seed(42))

        self.train_set = DatasetFromSubset(train_set, self.train_transforms)
        self.valid_set = DatasetFromSubset(valid_set, self.valid_transforms)

    def train(self, rank, lock):
        torch.manual_seed(self.args.seed + rank)
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
            self.train_epoch(epoch, train_loader, optimizer, scheduler, criterion, lock)


    def train_epoch(self, epoch, train_loader, optimizer, scheduler, criterion, lock):
        epoch_time = time.time()

        scheduler.step()
        self.model.train(True)
        pid = os.getpid()
        running_loss = 0.0
        train_running_corrects = 0

        loader_iter = iter(train_loader)
        for batch_idx in range(len(train_loader)):
            if self.shared_dataset.get_accesses(lock, batch_idx):
                inputs, labels = self.shared_dataset.get_batch(lock, batch_idx, pid)
            else:
                (inputs, labels, _) = next(loader_iter)
                inputs, labels = Variable(inputs.to(self.device)), Variable(labels.to(self.device))
                self.shared_dataset.set_batch(lock, inputs, labels, batch_idx, pid)

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

            self.shared_dataset.remove_batch(lock, batch_idx, pid)


        train_epoch_acc = train_running_corrects.double() / len(train_loader.dataset) * 100

        train_time = time.time() - epoch_time

        print(f"Training time: {train_time}s, Train accuracy: {train_epoch_acc}")

    def test(self):
        torch.manual_seed(self.args.seed)

        loader_valid = D.DataLoader(
            self.valid_set,
            batch_size=80,
            shuffle=True,
            num_workers=1,
            pin_memory=True,
            prefetch_factor=2,
        )

        criterion = torch.nn.CrossEntropyLoss()

        self.test_epoch(loader_valid, criterion)

    def test_epoch(self, loader_valid, criterion):
        self.model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target, _ in loader_valid:
                output = self.model(data.to(self.device))
                test_loss += criterion(output, target.to(self.device)).item()
                pred = output.max(1)[1]
                correct += pred.eq(target.to(self.device)).sum().item()

        test_loss /= len(loader_valid.dataset)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(loader_valid.dataset),
            100. * correct / len(loader_valid.dataset)))