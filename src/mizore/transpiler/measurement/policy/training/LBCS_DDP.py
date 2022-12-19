import torch.nn as nn
import torch.nn.functional as F
import torch

from tqdm import tqdm
from mizore.operators import QubitOperator
from mizore.testing.hamil import get_test_hamil
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from distQOdataset import DistQubitOperator
import argparse
import os 


device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda')

torch.manual_seed(3123)  # See https://arxiv.org/abs/2109.08203

pauli_map = {"X": 0, "Y": 1, "Z": 2}


def get_pword_tensor(pword, n_qubit):
    pauli_tensor = [[0.0, 0.0, 0.0] for _ in range(n_qubit)]
    for i_qubit, pauli in pword:
        pauli_tensor[i_qubit][pauli_map[pauli]] = 1.0
    return torch.tensor(pauli_tensor)


def get_operator_tensor(op: QubitOperator, n_qubit):
    coeffs = []
    pwords = []
    for pword, coeff in op.terms.items():
        pwords.append(get_pword_tensor(pword, n_qubit))
        coeffs.append(coeff)
    return torch.stack(pwords), torch.tensor(coeffs)


def get_no_zero_pauliwords(pauliwords):
    anti_qubit_mask = 1.0 - torch.sum(pauliwords, dim=-1)
    anti_qubit_mask: torch.tensor = anti_qubit_mask.unsqueeze(2)
    anti_qubit_mask = anti_qubit_mask.repeat(1, 1, 3)
    no_zero_pauliwords = pauliwords + anti_qubit_mask
    return no_zero_pauliwords


def get_shadow_coverage(heads, no_zero_pauliwords, head_ratios):
    shadow_coverage = torch.einsum("nqp, sqp -> nsq", no_zero_pauliwords, heads)
    coverage = torch.prod(shadow_coverage, dim=-1)
    coverage = torch.einsum("s, ns -> n", head_ratios, coverage)
    return coverage


# This loss is not average variance
def loss(heads, head_ratios, no_zero_pauliwords, coeffs):
    coverage = get_shadow_coverage(heads, no_zero_pauliwords, head_ratios)
    var = torch.sum(1.0 / coverage * (coeffs ** 2))
    return var


def polarization(heads, head_ratios):
    return torch.sum(torch.sum(torch.abs(heads - (1 / 3)), dim=(-1, -2)) * head_ratios)


class LargeLBCS(nn.Module):
    def __init__(self, init_head_ratios, init_heads):
        super(LargeLBCS, self).__init__()
        self.activator = torch.nn.Softplus()
        head_ratios = init_head_ratios
        heads = init_heads
        self.heads = torch.nn.Parameter(heads, requires_grad=True)
        self.head_ratios = torch.nn.Parameter(head_ratios, requires_grad=True)
        self.n_heads = len(heads)

    def forward(self, batch_pauli_tensor, batch_coeff):
        heads = self.activator(self.heads * 20)
        heads = F.normalize(heads, p=1.0, dim=-1)
        head_ratios = self.activator(self.head_ratios * 20)
        # head_ratios = F.normalize(head_ratios, p=1.0, dim=0)
        head_ratios = (F.normalize(head_ratios, p=1.0, dim=0) + (0.001 / self.n_heads)) / 1.001
        loss_val = loss(heads, head_ratios, batch_pauli_tensor, batch_coeff)
        return loss_val

# def train_model(gpu_num, n_head, mol_name, batch_size, n_step=2000):
def train_model(gpu_num, args):
    rank = args.nr * args.gpus + gpu_num
    torch.cuda.set_device(gpu_num)

    dist.init_process_group(
    	backend='nccl',
        init_method='env://',
    	world_size=args.world_size,
    	rank=rank
    )

    dset = DistQubitOperator(args.mol_name)
    n_qubit = dset.n_qubits

    train_sampler = torch.utils.data.distributed.DistributedSampler(
    	dset,
    	num_replicas=args.world_size,
    	rank=rank
    )

    train_loader = torch.utils.data.DataLoader(dataset=dset,
                                               batch_size=args.batch_size,
                                               shuffle=False,
                                               num_workers=0,
                                               pin_memory=True,
                                               sampler=train_sampler)


    head_ratios = torch.ones((args.n_head,)).to(gpu_num) * 10
    # head_ratios = (torch.flip(torch.range(start=1, end=n_head, step=1), dims=(-1,))*10).to(device)
    # head_ratios = (5+5*torch.rand((n_head,))).to(device)
    # heads = (5+5*torch.rand((n_head, n_qubit, 3))).to(device)
    heads = torch.ones((args.n_head, n_qubit, 3)).to(gpu_num) * 10
    model = LargeLBCS(head_ratios, heads)
    model.to(gpu_num)
    model = DDP(model, device_ids=[gpu_num])

    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-5)
    # optimizer = torch.optim.LBFGS(model.parameters())
    # optimizer = torch.optim.SGD(model.parameters(), lr=3)

    n_epoch = 0
    loss_in_epoch = []

    grad_mask = torch.zeros((args.n_head,))
    active_index = 0

    one_by_one = True
    normalize_grad = True
    with tqdm(range(args.epochs), ncols=100) as pbar:
        for step in pbar:
            epoch_loss = 0
            for i, batch in enumerate(train_loader):
                optimizer.zero_grad()
                pwords = batch["pword"]
                coeffs = batch["coeff"]
                # heads, head_ratios = model()

                batch_loss_val = model(pwords, coeffs)
                batch_loss_val.backward()
                epoch_loss += batch_loss_val
                params = model.named_parameters()

                if normalize_grad:
                    for name, param in params:
                        if name == "heads":
                            param.grad -= (torch.sum(param.grad, dim=-1) / 3).unsqueeze(-1)
                            pass
                        elif name == "head_ratios":
                            param.grad *= 0.01
                            param.grad -= (torch.sum(param.grad, dim=-1) / (args.n_head))
                        if n_epoch % 50 == -1:
                            print(name, torch.norm(param.grad))

                a = 1
                if one_by_one and active_index < a * args.n_head:
                    if n_epoch % 10 == 0:
                        grad_mask = torch.zeros((args.n_head,))
                        grad_mask[active_index % args.n_head] = 2.0
                        # second_active_index = (active_index + 1) % n_head
                        # grad_mask[second_active_index] = 1.0
                        active_index += 1
                        if active_index == a * args.n_head:
                            print("Pretrain finished")
                        # print(active_index)
                    params = model.named_parameters()
                    for name, param in params:
                        if name == "heads":
                            param.grad = torch.einsum("hqp, h->hqp", param.grad, grad_mask)
                        elif name == "head_ratios":
                            param.grad *= grad_mask
                        if n_epoch % 50 == -1:
                            print(1, name, torch.norm(param.grad))

                optimizer.step()
            
            loss_in_epoch.append(epoch_loss)
            if step % 3 == 0 and rank == 0:
                pbar.set_description('Var: {:.6f}'.format(epoch_loss))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--nodes', default=1,
                        type=int, metavar='N')
    parser.add_argument('-g', '--gpus', default=1, type=int,
                        help='number of gpus per node')
    parser.add_argument('-nr', '--nr', default=0, type=int,
                        help='ranking within the nodes')
    parser.add_argument('-batch_size', '--batch_size', default=128, type=int)
    parser.add_argument('-n_head', '--n_head', default=300, type=int)
    parser.add_argument('--epochs', default=50000, type=int, 
                        metavar='N',
                        help='number of total epochs to run')
    args = parser.parse_args()

    args.world_size = args.gpus * args.nodes
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12345'
    args.mol_name = "LiH_12_BK"
    # mp.spawn(train, nprocs=args.gpus, args=(args,))
    

    # mol_name = "NH3_30_BK"
    
    # train_model(n_head, hamil, 630 // 3 + 1, n_step=200000)
    mp.spawn(train_model, nprocs=args.gpus, args=(args,))

    # run script
    # python3 LBCS_DDP.py -n 1 -g 1 -nr 0 
