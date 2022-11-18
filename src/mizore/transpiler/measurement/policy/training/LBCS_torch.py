import torch.nn as nn
import torch.nn.functional as F
import torch

from tqdm import tqdm
import numpy as np

from mizore.operators import QubitOperator

torch.manual_seed(3407)  # See https://arxiv.org/abs/2109.08203

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


class LargeLBCS(nn.Module):
    def __init__(self, init_head_ratios, init_heads):
        super(LargeLBCS, self).__init__()
        self.activator = torch.nn.Softplus()
        head_ratios = init_head_ratios
        heads = init_heads
        self.heads = torch.nn.Parameter(heads, requires_grad=True)
        self.head_ratios = torch.nn.Parameter(head_ratios, requires_grad=True)

    def forward(self):
        heads = self.activator(self.heads)
        heads = F.normalize(heads, p=1.0, dim=-1)
        head_ratios = self.activator(self.head_ratios)
        head_ratios = F.normalize(head_ratios, p=1.0, dim=0)
        return heads, head_ratios


def train_model(n_head, hamil, batch_size, n_step=2000):
    n_qubit = hamil.n_qubit
    pauli_tensor, coeffs = get_operator_tensor(hamil, n_qubit)
    pauli_tensor = get_no_zero_pauliwords(pauli_tensor)
    n_pauliwords = len(coeffs)
    head_ratios = torch.tensor(np.random.uniform(size=(n_head,), low=5, high=10), dtype=torch.float)
    heads = torch.tensor(np.random.uniform(size=(n_head, n_qubit, 3), low=5, high=10), dtype=torch.float)

    model = LargeLBCS(head_ratios, heads)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-5)

    n_epoch = 0
    batch_n = 0
    with tqdm(range(n_step), ncols=100) as pbar:
        for step in pbar:
            optimizer.zero_grad()
            heads, head_ratios = model()
            if n_epoch % 30 == 0:
                total_loss = loss(heads, head_ratios, pauli_tensor, coeffs)
                pbar.set_description('Var: {:.6f}'.format(total_loss * (1 - 1 / (2 ** n_qubit + 1))))
            if n_epoch % 5 == 0:
                randperm = torch.randperm(n_pauliwords)
                pauli_tensor = pauli_tensor[randperm, :]
                coeffs = coeffs[randperm]
            batch_pauli_tensor = pauli_tensor[batch_n:batch_n + batch_size]
            batch_coeffs = coeffs[batch_n:batch_n + batch_size]
            batch_n += batch_size
            loss_val = loss(heads, head_ratios, batch_pauli_tensor, batch_coeffs)
            loss_val.backward()
            optimizer.step()
            if batch_n >= n_pauliwords:
                batch_n = 0
                n_epoch += 1


if __name__ == '__main__':
    from mizore.testing.hamil import get_test_hamil

    # jax.config.update('jax_platform_name', 'cuda')
    n_head = 300
    hamil, _ = get_test_hamil("mol", "LiH_12_BK").remove_constant()
    n_qubit = hamil.n_qubit

    train_model(n_head, hamil, 300, n_step=100000)
