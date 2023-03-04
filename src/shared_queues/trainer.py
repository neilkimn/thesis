import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils import data as D
from torch.autograd import Variable

import os
import time
from pathlib import Path

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

        with open(args.log_path + f"/{self.name}.csv", "a") as f:
            #f.write(f"{epoch},{train_acc},{valid_acc},{train_time},{train_corrects},{valid_corrects},{throughput}\n")
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

