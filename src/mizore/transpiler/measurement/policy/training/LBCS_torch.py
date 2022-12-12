import torch.nn as nn
import torch.nn.functional as F
import torch

from tqdm import tqdm
from mizore.operators import QubitOperator


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


def train_model(n_head, hamil, batch_size, n_step=2000):
    n_qubit = hamil.n_qubit
    pauli_tensor, coeffs = get_operator_tensor(hamil, n_qubit)
    pauli_tensor = get_no_zero_pauliwords(pauli_tensor)
    pauli_tensor = pauli_tensor.to(device)
    coeffs = coeffs.to(device)
    n_pauliwords = len(coeffs)

    head_ratios = torch.ones((n_head,)).to(device) * 10
    # head_ratios = (torch.flip(torch.range(start=1, end=n_head, step=1), dims=(-1,))*10).to(device)
    # head_ratios = (5+5*torch.rand((n_head,))).to(device)
    # heads = (5+5*torch.rand((n_head, n_qubit, 3))).to(device)
    heads = torch.ones((n_head, n_qubit, 3)).to(device) * 10
    model = LargeLBCS(head_ratios, heads).to(device)

    # https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html#torch.nn.parallel.DistributedDataParallel
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    # model = nn.DataParallel(model).to(device)

    # rank = 0
    # torch.distributed.init_process_group(backend='nccl', rank=rank, world_size=4)
    # model = torch.nn.parallel.DistributedDataParallel(LargeLBCS(head_ratios, heads), device_ids=[0,1],
    # output_device=rank).to(device)
    # model = LargeLBCS(head_ratios, heads).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-5)
    # optimizer = torch.optim.LBFGS(model.parameters())
    # optimizer = torch.optim.SGD(model.parameters(), lr=3)

    n_epoch = 0
    batch_n = 0
    loss_for_epoch = 0
    loss_in_epoch = []

    grad_mask = torch.zeros((n_head,))
    active_index = 0

    one_by_one = True
    normalize_grad = True
    with tqdm(range(n_step), ncols=100) as pbar:
        for step in pbar:
            optimizer.zero_grad()
            # heads, head_ratios = model()
            if n_epoch % 30 == 0:
                total_loss = torch.sum(model(pauli_tensor, coeffs)).cpu()
                pbar.set_description('Var: {:.6f}'.format(total_loss))
            if n_epoch % 5 == 0:
                randperm = torch.randperm(n_pauliwords)
                pauli_tensor = pauli_tensor[randperm, :]
                coeffs = coeffs[randperm]
            batch_pauli_tensor = pauli_tensor[batch_n:batch_n + batch_size]
            batch_coeffs = coeffs[batch_n:batch_n + batch_size]
            batch_n += batch_size
            # print(batch_coeffs.size())

            loss_val = model(batch_pauli_tensor, batch_coeffs)
            loss_val = torch.sum(loss_val)
            loss_in_epoch.append(loss_val.cpu())
            loss_val.backward()
            params = model.named_parameters()
            if normalize_grad:
                for name, param in params:
                    if name == "heads":
                        param.grad -= (torch.sum(param.grad, dim=-1) / 3).unsqueeze(-1)
                        pass
                    elif name == "head_ratios":
                        param.grad *= 0.01
                        param.grad -= (torch.sum(param.grad, dim=-1) / (n_head))
                    if n_epoch % 50 == -1:
                        print(name, torch.norm(param.grad))

            a = 1
            if one_by_one and active_index < a * n_head:
                if n_epoch % 10 == 0:
                    grad_mask = torch.zeros((n_head,))
                    grad_mask[active_index % n_head] = 2.0
                    # second_active_index = (active_index + 1) % n_head
                    # grad_mask[second_active_index] = 1.0
                    active_index += 1
                    if active_index == a * n_head:
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

            if batch_n >= n_pauliwords:
                batch_n = 0
                n_epoch += 1
                loss_for_epoch = sum(loss_in_epoch)
                loss_in_epoch = []


if __name__ == '__main__':
    from mizore.testing.hamil import get_test_hamil

    # mol_name = "NH3_30_BK"
    mol_name = "LiH_12_BK"
    # jax.config.update('jax_platform_name', 'cuda')
    n_head = 600
    hamil, _ = get_test_hamil("mol", mol_name).remove_constant()
    print("Hamiltonian contain {} terms".format(len(hamil.terms)))
    n_qubit = hamil.n_qubit

    train_model(n_head, hamil, 630 // 3 + 1, n_step=200000)
