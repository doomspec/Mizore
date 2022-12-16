import torch.nn as nn
import torch.nn.functional as F
import torch
import sys
import torchvision
import torchvision.transforms as transforms

from tqdm import tqdm
from mizore.operators import QubitOperator

import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from mizore.testing.hamil import get_test_hamil
import os
import argparse
import datetime
from distQOdataset import DistQubitOperator

pauli_map = {"X": 0, "Y": 1, "Z": 2}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--nodes', default=1, type=int, metavar='N')
    parser.add_argument('-g', '--gpus', default=1, type=int,
                        help='number of gpus per node')
    parser.add_argument('--epochs', default=2, type=int, metavar='N',
                        help='number of total epochs to run')
    args = parser.parse_args()

    args.world_size = args.gpus * args.nodes
    os.environ['MASTER_ADDR'] = '192.168.1.3'
    os.environ['MASTER_PORT'] = '8888'
    mp.spawn(train, nprocs=args.gpus, args=(args,))

def train(gpu, args):
    rank = args.nr * args.gpus + gpu
    dist.init_process_group(
    	backend='nccl',
        init_method='env://',
    	world_size=args.world_size,
    	rank=rank
    )

    torch.manual_seed(0)
    model = ConvNet()
    torch.cuda.set_device(gpu)
    model.cuda(gpu)
    batch_size = 100
    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(gpu)
    optimizer = torch.optim.SGD(model.parameters(), 1e-4)

    # Wrapper around our model to handle parallel training
    model = nn.parallel.DistributedDataParallel(model, device_ids=[gpu])

    # Data loading code
    train_dataset = torchvision.datasets.MNIST(root='./',
                                               train=True,
                                               transform=transforms.ToTensor(),
                                               download=True)
    
    # Sampler that takes care of the distribution of the batches such that
    # the data is not repeated in the iteration and sampled accordingly
    train_sampler = torch.utils.data.distributed.DistributedSampler(
    	train_dataset,
    	num_replicas=args.world_size,
    	rank=rank
    )
    
    # We pass in the train_sampler which can be used by the DataLoader
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=False,
                                               num_workers=0,
                                               pin_memory=True,
                                               sampler=train_sampler)

    start = datetime.now()
    total_step = len(train_loader)
    for epoch in range(args.epochs):
        for i, (images, labels) in enumerate(train_loader):
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (i + 1) % 100 == 0 and gpu == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(
                    epoch + 1, 
                    args.epochs, 
                    i + 1, 
                    total_step,
                    loss.item())
                   )
    if gpu == 0:
        print("Training complete in: " + str(datetime.now() - start))


def get_operator_tensor(op: QubitOperator, n_qubit):
    coeffs = []
    pwords = []
    for pword, coeff in op.terms.items():
        pwords.append(get_pword_tensor(pword, n_qubit))
        coeffs.append(coeff)
    return torch.stack(pwords), torch.tensor(coeffs)

def get_dist_operator_tensor(op: QubitOperator, n_qubit):
    dset = []

    for pword, coeff in op.terms.items():
        dset.append((get_pword_tensor(pword, n_qubit), coeff))
    
    return dset

def get_pword_tensor(pword, n_qubit):
    pauli_tensor = [[0.0, 0.0, 0.0] for _ in range(n_qubit)]
    for i_qubit, pauli in pword:
        pauli_tensor[i_qubit][pauli_map[pauli]] = 1.0
    return torch.tensor(pauli_tensor)

def test(gpu_num, args):
    mol_name = "LiH_12_BK"
    # jax.config.update('jax_platform_name', 'cuda')
    nodes = 1
    gpus = 4
    rank = (nodes - 1) * gpus + gpu_num
    n_head = 600
    dset = DistQubitOperator(mol_name)
    train_sampler = torch.utils.data.distributed.DistributedSampler(
    	dset,
    	num_replicas=args.world_size,
    	rank=rank
    )

    train_loader = torch.utils.data.DataLoader(dataset=dset,
                                               batch_size=1,
                                               shuffle=False,
                                               num_workers=0,
                                               pin_memory=True,
                                               sampler=train_sampler)
    
    idx_lst = []
    for i, data in enumerate(train_loader):
        idx_lst.append(i)
    
    print(idx_lst)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.world_size = 4
    mp.spawn(test, nprocs=args.world_size, args=(args,))
    # dataset = torchvision.datasets.MNIST(root='./',
    #     train=True,
    #     transform=transforms.ToTensor(),
    #     download=True)
    
    # print(dataset)