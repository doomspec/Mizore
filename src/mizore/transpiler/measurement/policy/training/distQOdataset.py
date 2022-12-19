from __future__ import print_function, division
import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from mizore.operators import QubitOperator
from mizore.testing.hamil import get_test_hamil
pauli_map = {"X": 0, "Y": 1, "Z": 2}

def get_pword_tensor(pword, n_qubit):
    pauli_tensor = [[0.0, 0.0, 0.0] for _ in range(n_qubit)]
    for i_qubit, pauli in pword:
        pauli_tensor[i_qubit][pauli_map[pauli]] = 1.0
    return torch.tensor(pauli_tensor)

def get_no_zero_pauliwords(pauliwords):
    anti_qubit_mask = 1.0 - torch.sum(pauliwords, dim=-1)
    anti_qubit_mask: torch.tensor = anti_qubit_mask.unsqueeze(2)
    anti_qubit_mask = anti_qubit_mask.repeat(1, 1, 3)
    no_zero_pauliwords = pauliwords + anti_qubit_mask
    return no_zero_pauliwords
class DistQubitOperator(Dataset):

    def __init__(self, mol_name):
        self.mol_name = mol_name
        self.hamil, _ = get_test_hamil("mol", mol_name).remove_constant()
        self.n_qubits = self.hamil.n_qubit
        self.pwords, self.coeff = self.get_operator_tensor(self.hamil, self.n_qubits)
        

    def get_operator_tensor(self, op: QubitOperator, n_qubit):
        coeffs = []
        pwords = []
        for pword, coeff in op.terms.items():
            pwords.append(get_pword_tensor(pword, n_qubit))
            coeffs.append(coeff)
        return get_no_zero_pauliwords(torch.stack(pwords)), torch.tensor(coeffs)
    
    def __len__(self):
        return self.coeff.size()[0]
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx[0]
        
        return {'pword': self.pwords[idx], 'coeff': self.coeff[idx]}
        

