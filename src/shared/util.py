from torch.multiprocessing import Value, Lock
import os

class MaxVal(object):
    def __init__(self, initval=0):
        self.val = Value('i', initval)
        self.lock = Lock()

    def value(self):
        with self.lock:
            return self.val.value
        
    def set_value(self, value):
        with self.lock:
            self.val.value = value

class Counter(object):
    def __init__(self, init_val=0):
        self.val = Value('i', init_val)
        self.init_val = init_val
        self.lock = Lock()

    def increment(self):
        with self.lock:
            self.val.value += 1

    def value(self):
        with self.lock:
            return self.val.value
        
    def reset(self):
        with self.lock:
            self.val.value = self.init_val

class AtomicInt(object):
    def __init__(self, initval=-1):
        self.val = Value('i', initval)
        self.lock = Lock()

    def value(self):
        with self.lock:
            return self.val.value
        
    def reset(self):
        with self.lock:
            self.val.value = -1

    def set_value(self, value):
        with self.lock:
            self.val.value = value

# TODO: Make this a shared object: https://stackoverflow.com/questions/3671666/sharing-a-complex-object-between-processes
class Logger(object):
    def __init__(self, args, pid=0, log_path=None, gpu_path=None):
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

        self.mps_train_time = 0
        self.mps_batch_time = 0
        self.mps_misc_time = 0 # TODO: Instead of rolling this into train time, save in distinct column?

    def set_mps_time(self, mps_time):
        self.mps_time = mps_time
        #self.mps_train_time = mps_time["train_time"]
        #self.mps_batch_time = mps_time["batch_time"]
        #self.mps_misc_time = mps_time["misc_time"]
    
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
        # If we have done some MPS-weight finding, we need to include the train and batch time from that in the total epoch time
        #if any((self.mps_train_time, self.mps_batch_time, self.mps_misc_time)):
        #    print(f"Adding time from MPS finding: {self.mps_train_time + self.mps_batch_time + self.mps_misc_time}")
        #    print(f"Train time: {self.mps_train_time}, batch time: {self.mps_batch_time}, misc time: {self.mps_misc_time}")
        #    self.train_time += (self.mps_train_time + self.mps_misc_time)
        #    self.val_time += self.mps_batch_time
        #    epoch_time += (self.mps_train_time + self.mps_batch_time + self.mps_misc_time)
        #    self.mps_train_time, self.mps_batch_time, self.mps_misc_time = 0,0,0
        if self.mps_time and epoch == 1:
            epoch_time += self.mps_time
            self.mps_time = 0

        print('{} Validation: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            self.pid, self.val_loss, self.val_correct, self.args.valid_dataset_len, self.val_acc))
        
        print(f"{self.pid} Epoch {epoch} end: {round(epoch_time,1)}s, Train accuracy: {round(train_acc,2)}")
        if self.args.log_dir:
            with open(self.log_path, "a") as f:
                f.write(f"{epoch},{train_acc},{self.val_acc},{self.train_time},{self.batch_time},{self.val_time},{epoch_time},{train_running_corrects},{self.val_correct}\n")
                os.system(f"nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader >> {self.gpu_path}")

        self.train_time = 0
        self.batch_time = 0

class MPSLogger(object):
    def __init__(self, log_path=None):
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.log_path, "w") as f:
            f.write("pid,iteration,index,model,qsize,incremented_index,weight,thread_percentage,convergence_count\n")

    def write_line(self, *args):
        line = ",".join(args)
        with open(self.log_path, "a") as f:
            f.write(f"{line}\n")

class MyQueue(object):
    def __init__(self, queue, index):
        self.queue = queue
        self.index = index


def write_debug_indices(indices, debug_indices_path, args):
    if args.debug_data_dir:
        with open(debug_indices_path, "a") as f:
            f.write(" ".join(list(map(str, indices.tolist()))))
            f.write("\n")


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

from torchvision import transforms
import torch

def get_transformations(dataset, input_size):
    if dataset == "compcars":
        train_transforms = transforms.Compose(
            [
                transforms.Resize((input_size, input_size)),
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
                transforms.Resize((input_size, input_size)),
                transforms.ToTensor(),
            ]
        )

    elif dataset == "imagenet":
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

        train_transforms = transforms.Compose([
                transforms.RandomResizedCrop(input_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])

        valid_transforms = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(input_size),
                transforms.ToTensor(),
                normalize,
            ])
    return train_transforms, valid_transforms

