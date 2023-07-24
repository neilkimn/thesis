import torch
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import torchvision.datasets as datasets

import torch.multiprocessing as mp

# import multiprocessing as mp
from torch.multiprocessing import Process, Queue, JoinableQueue
from torch.utils import data as D
from torch.autograd import Variable
import nvtx

from pathlib import Path
import shutil
import argparse
import random
import time
import os

from shared.dataset import CarDataset, DatasetFromSubset
from shared_queues.trainer import RCNNProcTrainer, CocoTrainer
from shared.util import Counter, get_transformations, write_debug_indices

import datetime
import os
import time

from torch_utils import presets
import torch
import torch.utils.data
import torchvision
import torchvision.models.detection
import torchvision.models.detection.mask_rcnn
import utils
from torch_utils.coco_utils import get_coco, get_coco_kp
from torch_utils.group_by_aspect_ratio import (
    create_aspect_ratio_groups,
    GroupedBatchSampler,
)
from torchvision.transforms import InterpolationMode
from torch_utils.transforms import SimpleCopyPaste


def copypaste_collate_fn(batch):
    copypaste = SimpleCopyPaste(
        blending=True, resize_interpolation=InterpolationMode.BILINEAR
    )
    return copypaste(*utils.collate_fn(batch))


def get_dataset(is_train, args):
    image_set = "train" if is_train else "val"
    paths = {
        "coco": (args.data_path, get_coco, 91),
        "coco_kp": (args.data_path, get_coco_kp, 2),
    }
    p, ds_fn, num_classes = paths[args.dataset]

    ds = ds_fn(p, image_set=image_set, transforms=get_transform(is_train, args))
    return ds, num_classes


def get_transform(is_train, args):
    if is_train:
        return presets.DetectionPresetTrain(
            data_augmentation=args.data_augmentation,
            backend=args.backend,
            use_v2=args.use_v2,
        )
    elif args.weights and args.test_only:
        weights = torchvision.models.get_weight(args.weights)
        trans = weights.transforms()
        return lambda img, target: (trans(img), target)
    else:
        return presets.DetectionPresetEval(backend=args.backend, use_v2=args.use_v2)


def get_args_parser(add_help=True):
    import argparse

    parser = argparse.ArgumentParser(
        description="PyTorch Detection Training", add_help=add_help
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--training-workers", type=int, default=1)
    parser.add_argument("--validation-workers", type=int, default=1)
    parser.add_argument("--prefetch-factor", type=int, default=1)
    parser.add_argument(
        "-a",
        "--arch",
        nargs="+",
        metavar="ARCH",
        help="model architecture (default: fasterrcnn_resnet50_fpn)",
        default="fasterrcnn_resnet50_fpn",
    )
    parser.add_argument("--num-processes", type=int, default=2)
    parser.add_argument(
        "--evaluate-only", action="store_true", help="whether to only run evaluation"
    )
    parser.add_argument(
        "--gpu-prefetch", action="store_true", help="whether to do GPU prefetching"
    )
    parser.add_argument(
        "--producer-per-worker",
        action="store_true",
        help="whether to have a producer for each worker",
    )
    parser.add_argument(
        "--debug_data_dir",
        metavar="DIR",
        nargs="?",
        default="",
        help="path to store data generated by dataloader",
    )
    parser.add_argument("--overwrite_debug_data", type=int, default=1)
    parser.add_argument(
        "--log_dir",
        metavar="LOG_DIR",
        nargs="?",
        default="",
        help="path to store training log",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        metavar="N",
        help="how many batches to wait before logging training status",
    )
    parser.add_argument(
        "--record_first_batch_time",
        action="store_true",
        help="Don't skip measuring time spent on first batch",
    )

    parser.add_argument(
        "--data-path", default="/datasets01/COCO/022719/", type=str, help="dataset path"
    )
    parser.add_argument("--dataset", default="coco", type=str, help="dataset name")
    parser.add_argument(
        "--device",
        default="cuda",
        type=str,
        help="device (Use cuda or cpu Default: cuda)",
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        default=2,
        type=int,
        help="images per gpu, the total batch size is $NGPU x batch_size",
    )
    parser.add_argument(
        "--epochs",
        default=26,
        type=int,
        metavar="N",
        help="number of total epochs to run",
    )
    parser.add_argument(
        "-j",
        "--workers",
        default=4,
        type=int,
        metavar="N",
        help="number of data loading workers (default: 4)",
    )
    parser.add_argument("--opt", default="sgd", type=str, help="optimizer")
    parser.add_argument(
        "--lr",
        default=0.02,
        type=float,
        help="initial learning rate, 0.02 is the default value for training on 8 gpus and 2 images_per_gpu",
    )
    parser.add_argument(
        "--momentum", default=0.9, type=float, metavar="M", help="momentum"
    )
    parser.add_argument(
        "--wd",
        "--weight-decay",
        default=1e-4,
        type=float,
        metavar="W",
        help="weight decay (default: 1e-4)",
        dest="weight_decay",
    )
    parser.add_argument(
        "--norm-weight-decay",
        default=None,
        type=float,
        help="weight decay for Normalization layers (default: None, same value as --wd)",
    )
    parser.add_argument(
        "--lr-scheduler",
        default="multisteplr",
        type=str,
        help="name of lr scheduler (default: multisteplr)",
    )
    parser.add_argument(
        "--lr-step-size",
        default=8,
        type=int,
        help="decrease lr every step-size epochs (multisteplr scheduler only)",
    )
    parser.add_argument(
        "--lr-steps",
        default=[16, 22],
        nargs="+",
        type=int,
        help="decrease lr every step-size epochs (multisteplr scheduler only)",
    )
    parser.add_argument(
        "--lr-gamma",
        default=0.1,
        type=float,
        help="decrease lr by a factor of lr-gamma (multisteplr scheduler only)",
    )
    parser.add_argument("--print-freq", default=20, type=int, help="print frequency")
    parser.add_argument(
        "--output-dir", default=".", type=str, help="path to save outputs"
    )
    parser.add_argument("--resume", default="", type=str, help="path of checkpoint")
    parser.add_argument("--start_epoch", default=0, type=int, help="start epoch")
    parser.add_argument("--aspect-ratio-group-factor", default=3, type=int)
    parser.add_argument(
        "--rpn-score-thresh",
        default=None,
        type=float,
        help="rpn score threshold for faster-rcnn",
    )
    parser.add_argument(
        "--trainable-backbone-layers",
        default=None,
        type=int,
        help="number of trainable layers of backbone",
    )
    parser.add_argument(
        "--data-augmentation",
        default="hflip",
        type=str,
        help="data augmentation policy (default: hflip)",
    )
    parser.add_argument(
        "--sync-bn",
        dest="sync_bn",
        help="Use sync batch norm",
        action="store_true",
    )
    parser.add_argument(
        "--test-only",
        dest="test_only",
        help="Only test the model",
        action="store_true",
    )

    parser.add_argument(
        "--use-deterministic-algorithms",
        action="store_true",
        help="Forces the use of deterministic algorithms only.",
    )

    # distributed training parameters
    parser.add_argument(
        "--world-size", default=1, type=int, help="number of distributed processes"
    )
    parser.add_argument(
        "--dist-url",
        default="env://",
        type=str,
        help="url used to set up distributed training",
    )
    parser.add_argument(
        "--weights", default=None, type=str, help="the weights enum name to load"
    )
    parser.add_argument(
        "--weights-backbone",
        default=None,
        type=str,
        help="the backbone weights enum name to load",
    )

    # Mixed precision training parameters
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Use torch.cuda.amp for mixed precision training",
    )

    # Use CopyPaste augmentation training parameter
    parser.add_argument(
        "--use-copypaste",
        action="store_true",
        help="Use CopyPaste data augmentation. Works only with data-augmentation='lsj'.",
    )

    parser.add_argument(
        "--backend",
        default="PIL",
        type=str.lower,
        help="PIL or tensor - case insensitive",
    )
    parser.add_argument("--use-v2", action="store_true", help="Use V2 transforms")

    return parser


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
        self.items_processed = 0

    def log_train_interval(
        self, idx, epoch, num_items, loss, items_processed, train_time, batch_time
    ):
        self.train_time = train_time
        self.batch_time = batch_time
        self.items_processed = items_processed

        if idx % self.args.log_interval == 0:
            print(
                "{}\tTrain Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.2f} Throughput [img/s]: {:.1f}".format(
                    self.pid,
                    epoch,
                    idx * num_items,
                    self.args.train_dataset_len,
                    100.0 * idx / self.args.train_loader_len,
                    loss,
                    items_processed / (train_time + batch_time),
                )
            )

    def log_validation(self, val_loss, val_correct, val_acc, val_time):
        self.val_time = val_time
        self.val_acc = val_acc
        self.val_loss = val_loss
        self.val_correct = val_correct

    def log_write_epoch_end(self, epoch, epoch_time, train_acc, train_running_corrects):
        print(
            "{} Validation: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
                self.pid,
                self.val_loss,
                self.val_correct,
                self.args.valid_dataset_len,
                self.val_acc,
            )
        )

        print(
            f"{self.pid} Epoch {epoch} end: {round(epoch_time,1)}s, Train accuracy: {round(train_acc,2)}"
        )
        if self.args.log_dir:
            with open(self.log_path, "a") as f:
                f.write(
                    f"{int(time.time())},{epoch},{train_acc},{self.val_acc},{self.train_time},{self.batch_time},{self.val_time},{epoch_time},{train_running_corrects},{self.val_correct},{self.items_processed/(self.train_time+self.batch_time)}\n"
                )
                os.system(
                    f"nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader >> {self.gpu_path}"
                )


def producer(
    loader,
    valid_loader,
    qs,
    qs_input,
    qs_labels,
    qs_bounce,
    device,
    args,
    producer_alive,
):
    pid = os.getpid()
    if args.seed:
        torch.manual_seed(args.seed)
    if args.debug_data_dir:
        if args.overwrite_debug_data:
            shutil.rmtree(args.debug_data_dir)
    debug_indices_path, debug_indices_val_path = None, None

    for epoch in range(1, args.epochs + 1):
        if args.debug_data_dir:
            debug_indices_path = (
                Path(args.debug_data_dir)
                / f"epoch_{epoch}"
                / f"{pid}_producer_indices.txt"
            )
            debug_indices_path.parent.mkdir(parents=True, exist_ok=True)

        if not args.evaluate_only:

            input_slots = []
            labels_slots = []

            for idx, (inputs_raw, labels_raw) in enumerate(loader):
                indices = None

                nvtx.push_range("producer memcpy")

                if idx < 20 and epoch == 1:
                    # prepopulate queues with tensor pointers
                    if args.gpu_prefetch:
                        inputs = list(image.to(device) for image in inputs_raw)
                    else:
                        inputs = inputs_raw

                    for input in inputs:
                        for q in qs_input:
                            q.queue.put(input)

                    # input_slots.extend(inputs)

                    if args.gpu_prefetch:
                        labels = [
                            v.to(device) if isinstance(v, torch.Tensor) else v
                            for t in labels_raw
                            for k, v in t.items()
                        ]
                    else:
                        labels = [v for t in labels_raw for k, v in t.items()]

                    for label in labels:
                        for q in qs_labels:
                            q.queue.put(label)

                    # labels_slots.extend(labels)

                nvtx.pop_range()

                # if args.gpu_prefetch:
                #     try:
                #         input_slots[offset * 2][:] = inputs[0]
                #         input_slots[offset * 2 + 1][:] = inputs[1]
                #     except Exception as e:
                #         pass

                for q in qs:
                    offset = idx % 20
                    # TODO: these try excepts *have* to go. The issue is that not all images are the same size. We should
                    # ensure that everything is the same size
                    # For images, that would mean resizing everything to some standard
                    if args.gpu_prefetch:
                        try:
                            input_slots[offset * 2][:] = inputs[0]
                            input_slots[offset * 2 + 1][:] = inputs[1]
                        except Exception as e:
                            pass

                        # labels = [v for t in labels_raw for k, v in t.items()]

                        # TODO: these try excepts *have* to go. The issue is that not all labels are the same size. We should
                        # ensure that everything is the same size
                        # Bounding boxes are a serious issue, because as far as I know there can be multiple boxes per image.
                        # Maybe check the literature what they do, or as interim we can limit the amount of boxes or something.
                        """ for i in range(12):
                            try:
                                labels_slots[offset*12+i][:] = labels[i]
                            except Exception as e:
                                pass
                                print(f"Index: {i}", e) """
                        """ try:
                            labels_slots[offset*12][:] = labels[0]
                            labels_slots[offset*12+1][:] = labels[1]
                            labels_slots[offset*12+2][:] = labels[2]
                            labels_slots[offset*12+3][:] = labels[3]
                            labels_slots[offset*12+4][:] = labels[4]
                            labels_slots[offset*12+5][:] = labels[5]
                            labels_slots[offset*12+6][:] = labels[6]
                            labels_slots[offset*12+7][:] = labels[7]
                            labels_slots[offset*12+8][:] = labels[8]
                            labels_slots[offset*12+9][:] = labels[9]
                            labels_slots[offset*12+10][:] = labels[10]
                            labels_slots[offset*12+11][:] = labels[11]
                        except Exception as e:
                            pass """

                    q.queue.put((idx, epoch, "train", indices))

                for q in qs_bounce:
                    while not q.queue.empty():
                        test = q.queue.get()
                    # test_b = q.queue.get()

        # TODO: fix
        # # end of training for epoch, switch to eval
        # if epoch > 2:
        #     for idx, (inputs, labels) in enumerate(valid_loader):
        #         indices = None
        #         if args.gpu_prefetch:
        #             inputs = list(image.to(device) for image in inputs)
        #             labels = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in labels]

        #         for q in qs:
        #             q.queue.put((idx, inputs, labels, epoch, "valid", indices))

        for q in qs:
            q.queue.put((0, epoch, "end", None))
    producer_alive.wait()


def worker(
    q, q_input, q_labels, q_bounce, model, args, producer_alive, finished_workers
):
    # affinity_mask = {i}
    # os.sched_setaffinity(0, affinity_mask)
    # torch.set_num_threads(1)
    pid = os.getpid()
    log_path, gpu_path = None, None
    if model.on_device == False:
        print(f"{pid}\tTraining model: {model.name}")
        model.send_model()
        model.start_epoch(epoch=0, data_loader_len=args.train_loader_len)
        if args.log_dir:
            log_path, gpu_path = model.init_log(pid)

    logger = Logger(args, pid, log_path, gpu_path)
    if not args.record_first_batch_time:
        print("Skipping recording batch time for first batch!")

    epochs_processed = 0
    epoch_start_time = time.time()

    train_time, val_time, batch_time, queue_time, items_processed = 0, 0, 0, 0, 0

    input_slots = [q_input.queue.get() for _ in range(40)]
    labels_slots = [q_labels.queue.get() for _ in range(240)]

    while True:
        pid = os.getpid()

        start = time.time()
        nvtx.push_range("worker get batch")

        idx, epoch, batch_type, indices = q.queue.get()
        queue_time += time.time() - start
        offset = idx % 20
        inputs = input_slots[offset * 2 : offset * 2 + 2]

        labels = [
            {
                "boxes": labels_slots[offset * 12],
                "labels": labels_slots[offset * 12 + 1],
                "masks": labels_slots[offset * 12 + 2],
                "image_id": labels_slots[offset * 12 + 3],
                "area": labels_slots[offset * 12 + 4],
                "iscrowd": labels_slots[offset * 12 + 5],
            },
            {
                "boxes": labels_slots[offset * 12 + 6],
                "labels": labels_slots[offset * 12 + 7],
                "masks": labels_slots[offset * 12 + 8],
                "image_id": labels_slots[offset * 12 + 9],
                "area": labels_slots[offset * 12 + 10],
                "iscrowd": labels_slots[offset * 12 + 11],
            },
        ]

        nvtx.pop_range()

        if batch_type in ("train", "valid"):
            if not args.gpu_prefetch:
                nvtx.push_range("worker memcpy")
                inputs = list(image.to(args.device) for image in inputs)

                for input in inputs:
                    q_bounce.queue.put(input)

                labels = [
                    {
                        k: v.to(args.device) if isinstance(v, torch.Tensor) else v
                        for k, v in t.items()
                    }
                    for t in labels
                ]
                nvtx.pop_range()

        batch_time += time.time() - start

        start = time.time()
        if batch_type == "train":
            nvtx.push_range("model forward")
            loss = model.forward(inputs, labels)
            nvtx.pop_range()
            items_processed += len(inputs)
            train_time += time.time() - start
            logger.log_train_interval(
                idx, epoch, len(inputs), loss, items_processed, train_time, batch_time
            )

        elif batch_type == "valid":

            val_loss, val_acc, val_correct = model.validate(inputs, labels)
            val_time += time.time() - start
            logger.log_validation(val_loss, val_correct, val_acc, val_time)

        elif batch_type == "end":
            train_epoch_acc, train_running_corrects = model.end_epoch()
            epoch_end_time = time.time() - epoch_start_time
            epoch_time = train_time + val_time + batch_time
            print(
                f"Train + val + batch time: {train_time}, {val_time}, {batch_time} ({queue_time}), total: {train_time+val_time+batch_time}"
            )
            logger.log_write_epoch_end(
                epoch, epoch_end_time, train_epoch_acc, train_running_corrects
            )
            train_time, val_time, batch_time, queue_time, items_processed = (
                0,
                0,
                0,
                0,
                0,
            )
            model.start_epoch(epoch + 1, args.train_loader_len)
            epoch_start_time = time.time()
        # nvtx.push_range("task done")
        # q.queue.task_done()
        # nvtx.pop_range()
        if batch_type == "end":
            epochs_processed += 1
            if epochs_processed == args.epochs:
                finished_workers.increment()
                if finished_workers.value() == args.num_processes:
                    producer_alive.set()
                break


def main(args):

    if args.seed is not None:
        print(f"Setting seed {args.seed}")
        random.seed(args.seed)
        torch.manual_seed(args.seed)

    mp.set_sharing_strategy("file_descriptor")
    mp.set_start_method("spawn", force=True)
    manager = mp.Manager()

    device = torch.device("cuda")

    if args.backend.lower() == "datapoint" and not args.use_v2:
        raise ValueError("Use --use-v2 if you want to use the datapoint backend.")

    if args.output_dir:
        utils.mkdir(args.output_dir)

    utils.init_distributed_mode(args)
    print(args)

    device = torch.device(args.device)

    if args.use_deterministic_algorithms:
        torch.use_deterministic_algorithms(True)

    # Data loading code
    print("Loading data")

    dataset, num_classes = get_dataset(is_train=True, args=args)
    dataset_test, _ = get_dataset(is_train=False, args=args)

    print("Creating data loaders")
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        test_sampler = torch.utils.data.distributed.DistributedSampler(
            dataset_test, shuffle=False
        )
    else:
        train_sampler = torch.utils.data.RandomSampler(dataset)
        test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    if args.aspect_ratio_group_factor >= 0:
        group_ids = create_aspect_ratio_groups(
            dataset, k=args.aspect_ratio_group_factor
        )
        train_batch_sampler = GroupedBatchSampler(
            train_sampler, group_ids, args.batch_size
        )
    else:
        train_batch_sampler = torch.utils.data.BatchSampler(
            train_sampler, args.batch_size, drop_last=True
        )

    train_collate_fn = utils.collate_fn
    if args.use_copypaste:
        if args.data_augmentation != "lsj":
            raise RuntimeError(
                "SimpleCopyPaste algorithm currently only supports the 'lsj' data augmentation policies"
            )

        train_collate_fn = copypaste_collate_fn

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_sampler=train_batch_sampler,
        num_workers=args.workers * args.num_processes,
        collate_fn=train_collate_fn,
    )

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=args.batch_size,
        sampler=test_sampler,
        num_workers=args.workers,
        collate_fn=utils.collate_fn,
    )

    args.train_dataset_len = len(data_loader.dataset)
    args.train_loader_len = len(data_loader)

    args.valid_dataset_len = len(data_loader_test.dataset)
    args.valid_loader_len = len(data_loader_test)

    args.lr_scheduler = args.lr_scheduler.lower()

    # trainer = CocoTrainer(args, model, device)
    print("Creating model")

    train_models = []
    for idx, arch in enumerate(args.arch):
        kwargs = {"trainable_backbone_layers": args.trainable_backbone_layers}
        if args.data_augmentation in ["multiscale", "lsj"]:
            kwargs["_skip_resize"] = True
        if "rcnn" in arch:
            if args.rpn_score_thresh is not None:
                kwargs["rpn_score_thresh"] = args.rpn_score_thresh
        model = torchvision.models.get_model(
            arch,
            weights=args.weights,
            weights_backbone=args.weights_backbone,
            num_classes=num_classes,
            **kwargs,
        )
        trainer = CocoTrainer(args, model, device, arch)
        if args.lr_scheduler == "multisteplr":
            epoch_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                trainer.optimizer, milestones=args.lr_steps, gamma=args.lr_gamma
            )
        elif args.lr_scheduler == "cosineannealinglr":
            epoch_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                trainer.optimizer, T_max=args.epochs
            )
        else:
            raise RuntimeError(
                f"Invalid lr scheduler '{args.lr_scheduler}'. Only MultiStepLR and CosineAnnealingLR are supported."
            )
        trainer.set_epoch_lr_scheduler(epoch_lr_scheduler)
        train_models.append(trainer)

    queues = []
    queues_input = []
    queues_labels = []
    queues_bounce = []
    for idx in range(args.num_processes):
        q = JoinableQueue(maxsize=20)
        q_input = JoinableQueue(maxsize=40)  # batch size 2, so 2*20
        q_labels = JoinableQueue(maxsize=240)  # 6 dict elements per, so 20*40
        q_bounce = JoinableQueue(maxsize=40)
        queue = MyQueue(q, idx)
        queue_input = MyQueue(q_input, idx)
        queue_labels = MyQueue(q_labels, idx)
        queue_bounce = MyQueue(q_bounce, idx)
        queues.append(queue)
        queues_input.append(queue_input)
        queues_labels.append(queue_labels)
        queues_bounce.append(queue_bounce)

    producer_alive = mp.Event()
    producers = []

    if args.producer_per_worker:
        for i in range(args.num_processes):
            data_loader = torch.utils.data.DataLoader(
                dataset,
                batch_sampler=train_batch_sampler,
                num_workers=args.workers,
                collate_fn=train_collate_fn,
            )

            data_loader_test = torch.utils.data.DataLoader(
                dataset_test,
                batch_size=1,
                sampler=test_sampler,
                num_workers=args.workers,
                collate_fn=utils.collate_fn,
            )
            p = Process(
                target=producer,
                args=(
                    (
                        data_loader,
                        data_loader_test,
                        [queues[i]],
                        [queues_input[i]],
                        [queues_labels[i]],
                        [queues_bounce[i]],
                        device,
                        args,
                        producer_alive,
                    )
                ),
            )
            producers.append(p)
            p.start()
    else:
        p = Process(
            target=producer,
            args=(
                (
                    data_loader,
                    data_loader_test,
                    queues,
                    queues_input,
                    queues_labels,
                    queues_bounce,
                    device,
                    args,
                    producer_alive,
                )
            ),
        )
        producers.append(p)
        p.start()

    args.device = device
    finished_workers = Counter(0)
    workers = []
    start_time = time.time()
    for i in range(args.num_processes):
        p = Process(
            target=worker,
            daemon=True,
            args=(
                (
                    queues[i],
                    queues_input[i],
                    queues_labels[i],
                    queues_bounce[i],
                    train_models[i],
                    args,
                    producer_alive,
                    finished_workers,
                )
            ),
        )
        workers.append(p)
        p.start()

    for p in workers:
        p.join()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Training time {total_time_str}")


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)
