# implement the clip loss for ddp in two ways, compare the two implementations, this script is for the first implementation by torch
# this should be tested on two nvidia gpus

import torch
import random

# set random seed for reproducibility
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)
random.seed(0)

# create data tensors
# x = torch.rand(4, 2) # stands for 8 samples with 5 features, such as 8 images, and for each gpu it has 4 images
x = torch.tensor([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]])
# y = torch.rand(4, 2) # texts for the images, 4 texts for each gpu
y = torch.tensor([[0.2, 0.3], [0.3, 0.6], [0.4, 0.9], [0.2, 0.5]])

# create model
linear_model = torch.nn.Linear(2, 2)

with torch.no_grad():
    linear_model.weight.fill_(1.0)  # 将所有权重设置为1
    linear_model.bias.fill_(0.0) 

# clip loss implemented by torch

from clip_loss import ClipLoss

# torch dataset
from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


custom_dataset = CustomDataset(x, y)
custom_dataloader = DataLoader(custom_dataset, batch_size=2, shuffle=False)


import os
import sys
import tempfile
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp

from torch.nn.parallel import DistributedDataParallel as DDP

# On Windows platform, the torch.distributed package only
# supports Gloo backend, FileStore and TcpStore.
# For FileStore, set init_method parameter in init_process_group
# to a local file. Example as follow:
# init_method="file:///f:/libtmp/some_file"
# dist.init_process_group(
#    "gloo",
#    rank=rank,
#    init_method=init_method,
#    world_size=world_size)
# For TcpStore, same way as on Linux.

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def demo_basic(rank, world_size):
    try:
        print(f"Running basic DDP example on rank {rank}.")
        setup(rank, world_size)

        # create model and move it to GPU with id rank
        model = linear_model.to(rank)

        ddp_model = DDP(model, device_ids=[rank])

        print(f"before, linear model params:")

        for name, param in ddp_model.named_parameters():
            print(f"Parameter name: {name}, Value: {param}\n")

        # loss_fn = nn.MSELoss()

        loss_fn = ClipLoss(local_loss=False, gather_with_grad=True, cache_labels=False, rank=rank, world_size=world_size, use_horovod=False, smoothing=0.)

        optimizer = optim.SGD(ddp_model.parameters(), lr=0.1)

        optimizer.zero_grad()

        x, y = next(iter(custom_dataloader)) # get the first batch

        # currently directly use the global x, y

        x = x.to(rank)
        print(f"input tensor on rank {rank} is {x}")
        outputs = ddp_model(x)
        print(f"output tensor on rank {rank} is {outputs}")
        labels = y.to(rank)
        print(f"label tensor on rank {rank} is {labels}")
        loss = loss_fn(outputs, labels)
        print(f"loss tensor on rank {rank} is {loss}")
        loss.backward()
        optimizer.step()
        
        print(f"after , linear model params:")

        for name, param in ddp_model.named_parameters():
            print(f"Parameter name: {name}, Value: {param}\n")

        cleanup()
        print(f"Finished running basic DDP example on rank {rank}.")
    except Exception as e:
        print(f"Error in basic DDP example on rank {rank}: {e}")
        raise e

def run_demo(demo_fn, world_size):
    mp.spawn(demo_fn,
             args=(world_size,),
             nprocs=world_size,
             join=True)


if __name__ == "__main__":
    n_gpus = torch.cuda.device_count()
    assert n_gpus >= 2, f"Requires at least 2 GPUs to run, but got {n_gpus}"
    world_size = n_gpus
    run_demo(demo_basic, world_size)

# usage:
# python -m torch.distributed.launch --nproc_per_node=4 --nnodes=1 \
# 	--master_port=12355 --use_env demo_tests/ddp_clip_loss_torch.py









