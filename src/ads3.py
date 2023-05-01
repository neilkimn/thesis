# Based on: https://github.com/mulin88/compcars

import warnings

warnings.filterwarnings("ignore")

import os
import time
from pathlib import Path

import torch
import torchvision
import torch.optim as optim

from torch.utils import data as D
from torch.optim import lr_scheduler
from torch.autograd import Variable




NUM_DATALOADER_WORKERS = 1

trainFilename = "data/train.txt"
root = Path(os.environ["DATA_PATH"]) / "compcars"
trainFilename = root / "train.txt"

def get_labels():
    ##############################
    # calculate the features size from paper's training dataset
    ##############################
    paperTrainFeatureSet = set()
    with open(trainFilename, newline="\n") as trainfile:
        for line in trainfile:
            feature = line.split("/")[1]
            paperTrainFeatureSet.add(feature)
    print("total train feature size: %s" % (len(paperTrainFeatureSet)))
    num_classes = len(paperTrainFeatureSet)
    print("num_classes:", num_classes)
    paperTrainFeatureList = sorted(paperTrainFeatureSet)

    return paperTrainFeatureList


def train_model(
    loader_train,
    loader_valid,
    model,
    criterion,
    optimizer,
    scheduler,
    log_name,
    epochs,
):
    dataset_sizes = {
        "train": len(loader_train.dataset),
        "valid": len(loader_valid.dataset),
    }
    print(
        f"Training on {len(loader_train.dataset)} samples. Validating on {len(loader_valid.dataset)} samples"
    )

    log_file = open(log_name, "w")
    log_file.write("epoch,train_acc,valid_acc,time,train_corr,valid_corr,throughput\n")

    for epoch in range(epochs):
        epoc_time = time.time()

        ### Train
        scheduler.step()
        model.train(True)
        running_loss = 0.0
        train_running_corrects = 0

        for inputs, labels in loader_train:
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())

            optimizer.zero_grad()

            outputs = model.forward(inputs)

            loss = criterion(outputs, labels)
            preds = torch.max(outputs, 1)[1]

            train_running_corrects += torch.sum(preds == labels.data)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        train_epoch_acc = train_running_corrects.double() / dataset_sizes["train"] * 100

        train_time = time.time() - epoc_time

        ### Validation
        model.train(False)
        running_loss = 0.0
        valid_running_corrects = 0

        for inputs, labels in loader_valid:
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())

            optimizer.zero_grad()

            outputs = model.forward(inputs)

            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs.data, 1)
            valid_running_corrects += torch.sum(preds == labels.data)

            running_loss += loss.item()

        valid_epoch_acc = valid_running_corrects.double() / dataset_sizes["valid"] * 100

        throughput = len(loader_train.dataset)/train_time

        print(
            "Epoch [{}/{}] train acc: {:.4f}% "
            "valid  acc: {:.4f}% Time: {:.0f}s train corr: {:d}  valid corr: {:d} throughput (img/s) {:.4f} ".format(
                epoch,
                epochs - 1,
                train_epoch_acc,
                valid_epoch_acc,
                (train_time),
                train_running_corrects,
                valid_running_corrects,
                throughput
            )
        )
        log_file.write(f"{epoch},{train_epoch_acc},{valid_epoch_acc},{train_time},{train_running_corrects},{valid_running_corrects},{throughput}\n")

    log_file.close()
    return model


def run_experiment(loader_train, loader_valid, log_name, epochs=10):
    if not torch.cuda.is_available():
        raise Exception("CUDA not available to Torch!")

    device = torch.device("cuda:0")
    net = torchvision.models.resnet18(pretrained="imagenet")

    num_ftrs = net.fc.in_features  # num_ftrs = 2048
    print("features", num_ftrs, "classes", 431)
    net.fc = torch.nn.Linear(num_ftrs, 431)

    if torch.cuda.device_count() > 1:
        net = torch.nn.DataParallel(net)

    net.cuda()

    criterion = torch.nn.CrossEntropyLoss()
    criterion.cuda()

    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
    my_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    if torch.cuda.device_count() > 1:
        net = torch.nn.DataParallel(net)
    net.cuda()

    start_time = time.time()
    model = train_model(
        loader_train,
        loader_valid,
        net,
        criterion,
        optimizer,
        my_scheduler,
        log_name,
        epochs=epochs,
    )
    print("Training time: {:10f} minutes".format((time.time() - start_time) / 60))