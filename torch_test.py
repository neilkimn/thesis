import torch
import time



# device = torch.device("cuda")

# tensor_a = torch.rand((2, 2)).to(device)


# print(tensor_a.data_ptr(), tensor_a)

# tensor_b = torch.rand((2,2))


# print(tensor_a.data_ptr(), tensor_a)
# print(tensor_b.data_ptr(), tensor_b)

# tensor_a[:] = tensor_b

# print(tensor_a.data_ptr(), tensor_a)
import torch.multiprocessing as mp

def reader(q):


    tensor = q.get()

    while True:
        print(tensor, tensor.data_ptr())
        time.sleep(0.05)

def writer(q):

    device = torch.device("cuda")
    tensor = torch.tensor((0,)).to(device)

    q.put(tensor)

    i = 0
    while True:
        i += 1
        new = torch.tensor((i,))
        tensor[:] = new
        print("INCREMENT!", tensor, tensor.data_ptr())
        time.sleep(0.1)


if __name__ == '__main__':
    num_processes = 2
    processes = []

    # new.share_memory_()

    q = mp.JoinableQueue(maxsize=20)

    # for rank in range(num_processes):
    p = mp.Process(target=writer, args=(q,))
    p.start()
    processes.append(p)

    p = mp.Process(target=reader, args=(q,))
    p.start()
    processes.append(p)

    # i = 0
